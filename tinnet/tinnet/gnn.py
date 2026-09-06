"""Shared crystal-graph convolutional network used in front of a physics head.

The network is theory-agnostic: it maps a crystal graph to ``n_out`` raw
outputs per *row*, where a row is a whole structure (``readout='mean'``), a
selected site (``readout='site'``), or every atom (``readout='atom'``).  A
:class:`~tinnet.tinnet.physics.base.PhysicsModel` then interprets the rows.

Parameter names match the published TinNet checkpoints, so the same class
loads the pretrained adsorption-energy (``readout='mean'``) and
cohesive-energy (``readout='atom'``) ensembles.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """CGCNN gated graph convolution with an optional neighbor padding mask."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, padding_filter=None):
        """
        atom_in_fea : (N, atom_fea_len)
        nbr_fea     : (N, M, nbr_fea_len)
        nbr_fea_idx : (N, M) long
        padding_filter : optional (N, M) 0/1 mask of real neighbors
        """
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len), atom_nbr_fea, nbr_fea], dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea).view(-1, self.atom_fea_len * 2)
        if padding_filter is None:
            total_gated_fea = self.bn1(total_gated_fea)
        else:
            real = torch.where(padding_filter.reshape(-1) == 1)[0]
            total_gated_fea[real] = self.bn1(total_gated_fea[real])
        total_gated_fea = total_gated_fea.view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        if padding_filter is None:
            nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        else:
            nbr_sumed = torch.sum(nbr_filter * nbr_core * padding_filter[:, :, None], dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        return self.softplus2(atom_in_fea + nbr_sumed)


class CrystalGraphConvNet(nn.Module):
    """CGCNN encoder with a configurable readout.

    Parameters
    ----------
    orig_atom_fea_len, nbr_fea_len
        Input atom / bond feature sizes.
    atom_fea_len, n_conv, h_fea_len, n_h
        Hidden width, number of graph convolutions, MLP width and depth.
    n_out
        Number of raw outputs per row (``physics.n_latent``).
    readout
        ``'mean'`` averages atom features per structure (one row per structure),
        ``'site'`` selects the feature of ``site_idx`` in each structure,
        ``'atom'`` keeps one row per atom.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3,
                 h_fea_len=128, n_h=1, n_out=1, readout="mean", **_ignored):
        super().__init__()
        if readout not in ("mean", "site", "atom"):
            raise ValueError("readout must be 'mean', 'site' or 'atom'.")
        self.readout = readout
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, n_out)

    def encode(self, atom_fea, nbr_fea, nbr_fea_idx, padding_filter=None):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx, padding_filter)
        return atom_fea

    def pool(self, atom_fea, crystal_atom_idx, site_idx=None):
        if self.readout == "atom":
            return atom_fea
        if crystal_atom_idx is None:
            raise ValueError("crystal_atom_idx is required for 'mean' and 'site' readouts.")
        assert sum(len(idx_map) for idx_map in crystal_atom_idx) == atom_fea.shape[0]
        if self.readout == "mean":
            rows = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        else:
            if site_idx is None:
                raise ValueError("site_idx is required for the 'site' readout.")
            rows = [atom_fea[idx_map[int(site)]][None, :] for idx_map, site in zip(crystal_atom_idx, site_idx)]
        return torch.cat(rows, dim=0)

    def head(self, fea):
        fea = self.conv_to_fc(self.conv_to_fc_softplus(fea))
        fea = self.conv_to_fc_softplus(fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                fea = softplus(fc(fea))
        return self.fc_out(fea)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx=None, site_idx=None, padding_filter=None):
        """Return ``(rows, n_out)`` raw outputs for the physics head."""
        atom_fea = self.encode(atom_fea, nbr_fea, nbr_fea_idx, padding_filter)
        return self.head(self.pool(atom_fea, crystal_atom_idx, site_idx))
