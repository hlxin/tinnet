"""Shared OC20-oriented energy model with conservative atomic forces.

This is a trainable research architecture, not a pretrained OC20 predictor.
It accepts variable molecules and arbitrary binding neighborhoods. Radial
message passing provides rotation-invariant energies and covariant forces;
it does not represent explicit directional orbitals or orbital mixing.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .physics.shared_adsorption import SharedAdsorptionPhysics


def _mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


class SharedAdsorptionModel(nn.Module):
    """Shared atomic encoder plus independent effective adsorption channels.

    Local atomic energies accommodate intramolecular/slab deformation and
    effects missing from the theory. Their partition from the theory terms
    is not uniquely identifiable from energy/force supervision alone.
    No adsorbate ID, manually assigned orbital list, or atop index is used.
    """

    def __init__(
        self,
        hidden=64,
        n_layers=3,
        n_rbf=24,
        n_channels=2,
        cutoff=6.0,
        n_grid=301,
        energy_reference="unspecified",
    ):
        super().__init__()
        if min(hidden, n_layers, n_channels) < 1 or n_rbf < 2:
            raise ValueError(
                "Positive layer/channel sizes and at least two radial basis functions are required."
            )
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be positive and finite.")
        self.config = dict(
            hidden=hidden,
            n_layers=n_layers,
            n_rbf=n_rbf,
            n_channels=n_channels,
            cutoff=cutoff,
            n_grid=n_grid,
            energy_reference=energy_reference,
        )
        self.cutoff = cutoff
        self.n_channels = n_channels
        self.energy_reference = energy_reference
        self.embedding = nn.Embedding(119, hidden)
        self.role_embedding = nn.Embedding(2, hidden)
        self.register_buffer("rbf_centers", torch.linspace(0, cutoff, n_rbf))
        self.messages = nn.ModuleList(
            [_mlp(2 * hidden + n_rbf, hidden, hidden) for _ in range(n_layers)]
        )
        self.updates = nn.ModuleList(
            [_mlp(2 * hidden, hidden, hidden) for _ in range(n_layers)]
        )
        self.local_head = _mlp(hidden, hidden, 1)
        self.channel_head = _mlp(hidden, hidden, 3 * n_channels)
        self.surface_head = _mlp(hidden, hidden, 2)
        self.coupling_head = _mlp(2 * hidden + n_rbf, hidden, n_channels)
        self.physics = SharedAdsorptionPhysics(n_grid=n_grid)

    def _energy(self, graph, positions):
        if abs(graph["cutoff"] - self.cutoff) > 1e-8:
            raise ValueError("Graph and model cutoff must match; rebuild the graph.")
        src, dst = graph["edge_index"]
        offsets = torch.einsum(
            "ei,eij->ej", graph["edge_shifts"], graph["cell"][graph["batch"][src]]
        )
        vectors = positions[dst] - positions[src] + offsets
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        u = (distance / self.cutoff).clamp(max=1.0)
        # Quintic cutoff: value, first derivative, and second derivative vanish.
        envelope = 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5
        rbf = torch.exp(
            -(
                (
                    (distance[:, None] - self.rbf_centers)
                    / (self.cutoff / (len(self.rbf_centers) - 1))
                )
                ** 2
            )
        )
        ads = graph["adsorbate_mask"]
        h = self.embedding(graph["numbers"]) + self.role_embedding(ads.long())
        for message, update in zip(self.messages, self.updates):
            pair = torch.cat((h[src], h[dst], rbf), dim=-1)
            m = message(pair) * envelope[:, None]
            aggregated = torch.zeros_like(h).index_add(0, src, m)
            h = h + update(torch.cat((h, aggregated), dim=-1))
        n_graphs = len(graph["ptr"]) - 1
        local = h.new_zeros(n_graphs).index_add(
            0, graph["batch"], self.local_head(h).squeeze(-1)
        )

        # Directed edges from every adsorbate atom to all nearby slab images.
        cross = ads[src] & ~ads[dst]
        a, s = src[cross], dst[cross]
        pair = torch.cat((h[a], h[s], rbf[cross]), dim=-1)
        weights = F.softplus(self.coupling_head(pair)) * envelope[cross, None].square()
        weights = weights / self.n_channels
        coupling = h.new_zeros((len(h), self.n_channels)).index_add(0, a, weights)
        surface = self.surface_head(h[s])
        weighted_surface = h.new_zeros((len(h), self.n_channels, 2)).index_add(
            0, a, weights[:, :, None] * surface[:, None, :]
        )
        # Smooth denominator keeps derivatives finite as a contact disappears.
        band = weighted_surface[ads] / (coupling[ads, :, None] + 1e-6)
        orbital = self.channel_head(h[ads]).reshape(-1, self.n_channels, 3)
        latent = torch.stack(
            (
                orbital[..., 0],
                coupling[ads],
                orbital[..., 1],
                band[..., 0],
                band[..., 1],
                orbital[..., 2],
            ),
            dim=-1,
        )
        energy, parameters = self.physics.forward(
            latent, {"local_energy": local, "adsorbate_batch": graph["batch"][ads]}
        )
        # Keeps the zero-force, edgeless case connected to positions for autograd.
        energy = energy + positions.sum() * 0.0
        return {
            "energy": energy,
            "parameters": parameters,
            "channel_parameters": self.physics.channel_parameters(latent),
            "adsorbate_batch": graph["batch"][ads],
        }

    def forward(self, graph, *, compute_forces=False, create_graph=False):
        """Return energies (B,), optional forces (N,3), and theory contributions.

        Set create_graph=True for training on forces (second derivatives).
        Physical forces are returned for all atoms; the loss/ASE constraints
        decide which atoms to train on or move. Rebuild graphs after motion.
        """
        with torch.enable_grad() if compute_forces else nullcontext():
            positions = graph["positions"]
            if compute_forces and not positions.requires_grad:
                positions = positions.detach().clone().requires_grad_(True)
            result = self._energy(graph, positions)
            if compute_forces:
                result["forces"] = -torch.autograd.grad(
                    result["energy"].sum(),
                    positions,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                )[0]
        return result
