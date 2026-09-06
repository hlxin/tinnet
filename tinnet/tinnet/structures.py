"""Complete adsorbate/slab configurations and periodic radius graphs.

OC20 tags are 0 (fixed slab), 1 (free slab), and 2 (adsorbate).
Alternatively supply explicit masks; chemical symbols never determine roles.
Neighbor indices are discrete, but distances are recomputed in torch inside
the model. No stored distances are used when differentiating the energy.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list


def _mask(values, n, name):
    values = np.asarray(values)
    if values.shape != (n,) or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must be a Boolean mask with one entry per atom.")
    return values.astype(bool)


def configuration_graph(
    atoms, *, adsorbate_mask=None, fixed_mask=None, cutoff=6.0, dtype=torch.float32
):
    """Build one graph without removing, inserting, or repositioning atoms.

    All periodic images inside the cutoff are retained. There is no top-k
    truncation, which would introduce discontinuities during relaxation.
    Explicit masks take precedence over tags. ASE FixAtoms constraints are
    respected when no fixed mask is supplied.
    """
    n = len(atoms)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be positive and finite.")
    if (
        not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
    ):
        raise ValueError("Positions and cell must be finite.")
    if n == 0 or np.any((atoms.numbers < 1) | (atoms.numbers > 118)):
        raise ValueError("Expected atoms with atomic numbers between 1 and 118.")
    use_tags = adsorbate_mask is None
    tags = atoms.get_tags()
    if use_tags:
        if not np.isin(tags, [0, 1, 2]).all() or not np.any(tags == 2):
            raise ValueError(
                "Supply adsorbate_mask or OC20 tags (0/1 slab, 2 adsorbate)."
            )
        adsorbate_mask = tags == 2
    adsorbate_mask = _mask(adsorbate_mask, n, "adsorbate_mask")
    if not adsorbate_mask.any() or adsorbate_mask.all():
        raise ValueError("A configuration needs both adsorbate and slab atoms.")
    if not np.isin(atoms.numbers[adsorbate_mask], [1, 6, 7, 8]).all():
        raise ValueError("This OC20 model supports H, C, N, and O adsorbate atoms.")
    if fixed_mask is None:
        fixed_mask = tags == 0 if use_tags else np.zeros(n, dtype=bool)
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_mask[constraint.get_indices()] = True
    fixed_mask = _mask(fixed_mask, n, "fixed_mask")
    i, j, shifts = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
    return {
        "numbers": torch.as_tensor(atoms.numbers.copy(), dtype=torch.long),
        "positions": torch.as_tensor(atoms.positions.copy(), dtype=dtype),
        "cell": torch.as_tensor(atoms.cell.array.copy(), dtype=dtype)[None],
        "edge_index": torch.as_tensor(np.stack((i, j)), dtype=torch.long),
        "edge_shifts": torch.as_tensor(shifts, dtype=dtype),
        "adsorbate_mask": torch.as_tensor(adsorbate_mask),
        "fixed_mask": torch.as_tensor(fixed_mask),
        "batch": torch.zeros(n, dtype=torch.long),
        "ptr": torch.tensor([0, n], dtype=torch.long),
        "cutoff": float(cutoff),
    }


def collate_configurations(samples):
    """Batch graph/energy/forces samples without padding atoms or channels.

    Each sample contains ``graph`` and optional ``energy`` (eV), ``forces``
    (eV/Angstrom), ``energy_reference``, ``sid``, and ``fid``. A label must be
    supplied for every sample or none. Keep all frames of a sid in one split.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    graphs = [s["graph"] for s in samples]
    cutoff = graphs[0]["cutoff"]
    if any(g["cutoff"] != cutoff or len(g["ptr"]) != 2 for g in graphs):
        raise ValueError("Collate single configurations with the same cutoff.")
    references = {s.get("energy_reference", "unspecified") for s in samples}
    if len(references) != 1:
        raise ValueError("Do not mix energy reference conventions in a batch.")
    sizes = [len(g["numbers"]) for g in graphs]
    offsets = np.cumsum([0] + sizes)
    graph = {
        key: torch.cat([g[key] for g in graphs])
        for key in (
            "numbers",
            "positions",
            "cell",
            "edge_shifts",
            "adsorbate_mask",
            "fixed_mask",
        )
    }
    graph["edge_index"] = torch.cat(
        [g["edge_index"] + int(o) for g, o in zip(graphs, offsets)], dim=1
    )
    graph["batch"] = torch.repeat_interleave(
        torch.arange(len(graphs)), torch.tensor(sizes)
    )
    graph["ptr"] = torch.as_tensor(offsets, dtype=torch.long)
    graph["cutoff"] = cutoff
    batch = {
        "graph": graph,
        "energy_reference": references.pop(),
        "sid": [s.get("sid") for s in samples],
        "fid": [s.get("fid") for s in samples],
    }
    for key in ("energy", "forces"):
        present = [key in s for s in samples]
        if any(present) and not all(present):
            raise ValueError(f"Missing {key} labels in part of the batch.")
        if all(present):
            values = [
                torch.as_tensor(s[key], dtype=graph["positions"].dtype) for s in samples
            ]
            if key == "energy":
                if any(v.numel() != 1 for v in values):
                    raise ValueError("Expected one energy per configuration.")
                batch[key] = torch.stack([v.reshape(()) for v in values])
            else:
                if any(v.shape != (n, 3) for v, n in zip(values, sizes)):
                    raise ValueError("Forces must have shape (n_atoms, 3).")
                batch[key] = torch.cat(values)
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f"Non-finite {key} labels.")
    return batch


def move_batch(batch, device):
    """Move tensors in a graph or training batch, retaining metadata."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    return batch
