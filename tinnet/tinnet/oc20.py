"""OC20 S2EF adapters. Legacy LMDBs contain pickles: use trusted sources.

Energy reference conventions must be declared. Optional reference subtraction
is allowed only for total-energy labels. Stored neighbor graphs are rebuilt.
"""

from __future__ import annotations

import bisect
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from torch.utils.data import Dataset

from .structures import configuration_graph


def _field(record, name, default=None):
    if isinstance(record, dict):
        return record.get(name, default)
    # Early PyG pickles stored attributes directly, before _store existed.
    attrs = vars(record)
    if "_store" not in attrs:
        return attrs.get(name, default)
    return getattr(record, name, default)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _identifier(value):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value.item()
    return value


def oc20_sample(
    record,
    *,
    energy_reference,
    cutoff=6.0,
    reference_energy=None,
    dtype=torch.float32,
    energy_key="y",
):
    """Convert a dict or PyG S2EF record to graph, labels, and system/frame IDs.

    Required: atomic_numbers, pos, cell, tags. Optional: fixed, pbc, force, y,
    sid, fid. Periodicity defaults to True, matching OC20. y_relaxed is never
    substituted for the supplied geometry's energy. energy_reference describes
    the incoming labels: 'oc20-referenced', 'total', or 'synthetic'.
    """
    if energy_reference not in ("oc20-referenced", "total", "synthetic"):
        raise ValueError(
            "Declare energy_reference as oc20-referenced, total, or synthetic."
        )
    missing = [
        n
        for n in ("atomic_numbers", "pos", "cell", "tags")
        if _field(record, n) is None
    ]
    if missing:
        raise ValueError(f"Missing OC20 geometry fields: {missing}")
    tags = _numpy(_field(record, "tags"))
    if not np.isin(tags, [0, 1, 2]).all():
        raise ValueError(
            "OC20 tags must be 0 (fixed slab), 1 (free slab), or 2 (adsorbate)."
        )
    atoms = Atoms(
        numbers=_numpy(_field(record, "atomic_numbers")),
        positions=_numpy(_field(record, "pos")),
        cell=_numpy(_field(record, "cell")).reshape(3, 3),
        pbc=_numpy(_field(record, "pbc", [True, True, True])).reshape(-1),
        tags=tags,
    )
    fixed = _field(record, "fixed")
    graph = configuration_graph(
        atoms,
        fixed_mask=None if fixed is None else _numpy(fixed),
        cutoff=cutoff,
        dtype=dtype,
    )
    sample = {
        "graph": graph,
        "energy_reference": energy_reference,
        "sid": _identifier(_field(record, "sid")),
        "fid": _identifier(_field(record, "fid")),
    }
    energy = _field(record, energy_key)
    if energy is not None:
        energy = torch.as_tensor(_numpy(energy), dtype=dtype)
        if energy.numel() != 1 or not torch.isfinite(energy).all():
            raise ValueError("Expected one finite S2EF energy per frame.")
        sample["energy"] = energy.reshape(())
    elif _field(record, "y_relaxed") is not None:
        raise ValueError(
            "This is an IS2RE record; S2EF needs labels for the supplied geometry."
        )
    if reference_energy is not None:
        if energy_reference != "total":
            raise ValueError(
                "Reference subtraction requires total energies; labels may already be referenced."
            )
        if energy is None or not np.isfinite(reference_energy):
            raise ValueError(
                "Reference subtraction needs a finite energy and reference."
            )
        sample["energy"] = sample["energy"] - float(reference_energy)
        sample["energy_reference"] = "oc20-referenced"
    force = _field(record, "force")
    if force is not None:
        force = torch.as_tensor(_numpy(force), dtype=dtype)
        if force.shape != (len(atoms), 3) or not torch.isfinite(force).all():
            raise ValueError("Expected finite force labels of shape (n_atoms, 3).")
        sample["forces"] = force
    return sample


class OC20LmdbDataset(Dataset):
    """Lazy read-only legacy LMDB files/shards, with worker-local handles.

    reference_energies optionally maps sid (integer, string, or 'random<sid>')
    to slab+gas reference energy. Requires the 'oc20' installation extras.
    """

    def __init__(self, path, *, energy_reference, cutoff=6.0, reference_energies=None):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError(
                "Install the OC20 extras: pip install -e '.[oc20]'"
            ) from exc
        if energy_reference not in ("oc20-referenced", "total", "synthetic"):
            raise ValueError("Declare the dataset's energy reference convention.")
        if reference_energies is not None and energy_reference != "total":
            raise ValueError("Only total labels may have references subtracted.")
        self.energy_reference = energy_reference
        self.reference_energies = reference_energies
        self.cutoff = cutoff
        path = Path(path)
        self.paths = (
            [path]
            if path.is_file() or (path / "data.mdb").is_file()
            else sorted(path.glob("*.lmdb"))
        )
        if not self.paths:
            raise FileNotFoundError(f"No LMDB files found at {path}")
        self.ends = []
        total = 0
        for shard in self.paths:
            with lmdb.open(
                str(shard), subdir=shard.is_dir(), readonly=True, lock=False
            ) as env:
                with env.begin() as txn:
                    length = txn.get(b"length")
                    size = (
                        int(pickle.loads(length)) if length else txn.stat()["entries"]
                    )
                    if size and txn.get(str(size - 1).encode()) is None:
                        raise ValueError(
                            f"Expected contiguous numeric record keys in {shard}"
                        )
                    total += size
                    self.ends.append(total)
        self._envs = {}
        self._pid = os.getpid()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        import lmdb

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard = bisect.bisect_right(self.ends, index)
        local_index = index - (self.ends[shard - 1] if shard else 0)
        if shard not in self._envs:
            path = self.paths[shard]
            self._envs[shard] = lmdb.open(
                str(path),
                subdir=path.is_dir(),
                readonly=True,
                lock=False,
                readahead=False,
            )
        with self._envs[shard].begin() as txn:
            record = pickle.loads(txn.get(str(local_index).encode()))
        reference = None
        if self.reference_energies is not None:
            sid = _identifier(_field(record, "sid"))
            for key in (sid, str(sid), f"random{sid}"):
                if key in self.reference_energies:
                    reference = self.reference_energies[key]
                    break
            if reference is None:
                raise KeyError(f"Missing reference energy for sid={sid}")
        return oc20_sample(
            record,
            energy_reference=self.energy_reference,
            cutoff=self.cutoff,
            reference_energy=reference,
        )

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_envs"] = {}
        return state

    def __del__(self):
        if hasattr(self, "_envs"):
            self.close()
