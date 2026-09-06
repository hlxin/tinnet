#!/usr/bin/env python
"""Train the shared energy/force model on trusted OC20 LMDBs or a spring demo.

Run `python examples/train_shared_adsorption.py --help` after editable install.
The demo and 100-record OC20 tutorial validate plumbing, not scientific accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from torch.utils.data import DataLoader, Subset

from tinnet.tinnet.oc20 import OC20LmdbDataset
from tinnet.tinnet.shared_adsorption import SharedAdsorptionModel
from tinnet.tinnet.s2ef import evaluate_s2ef, fit_s2ef, save_shared_checkpoint
from tinnet.tinnet.structures import collate_configurations, configuration_graph


def spring_samples(cutoff):
    """Analytic, deliberately non-DFT labels for two adsorbates on a fixed atom."""
    samples = []
    for symbol in ("H", "O"):
        for r in np.linspace(1.5, 2.6, 8):
            atoms = Atoms(["Pt", symbol], positions=[[0, 0, 0], [0, 0, r]], tags=[0, 2])
            force = np.array([[0, 0, 1.2 * (r - 2.0)], [0, 0, -1.2 * (r - 2.0)]])
            samples.append(
                {
                    "graph": configuration_graph(atoms, cutoff=cutoff),
                    "energy": -1.5 + 0.6 * (r - 2.0) ** 2,
                    "forces": force,
                    "energy_reference": "synthetic",
                }
            )
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument(
        "--data", type=Path, help="Trusted legacy OC20 S2EF LMDB file/shard directory"
    )
    parser.add_argument(
        "--validation-data", type=Path, help="Separate official validation split"
    )
    parser.add_argument("--energy-reference", choices=["oc20-referenced", "total"])
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/shared-adsorption/model.pt")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, help="Optional maximum number of training records"
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--n-grid", type=int, default=301)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.data and args.energy_reference is None:
        parser.error(
            "Declare --energy-reference; the adapter cannot infer whether labels were referenced."
        )
    if args.demo and (args.validation_data or args.energy_reference):
        parser.error(
            "--demo uses synthetic labels and does not take an OC20 validation path or reference."
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    reference = "synthetic" if args.demo else args.energy_reference
    dataset = (
        spring_samples(args.cutoff)
        if args.demo
        else OC20LmdbDataset(args.data, energy_reference=reference, cutoff=args.cutoff)
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_configurations,
    )
    validation = None
    if args.validation_data:
        val_data = OC20LmdbDataset(
            args.validation_data, energy_reference=reference, cutoff=args.cutoff
        )
        validation = DataLoader(
            val_data, batch_size=args.batch_size, collate_fn=collate_configurations
        )
    model = SharedAdsorptionModel(
        hidden=args.hidden,
        n_layers=args.layers,
        n_channels=args.channels,
        cutoff=args.cutoff,
        n_grid=args.n_grid,
        energy_reference=reference,
    )
    before = evaluate_s2ef(
        model, loader, device=args.device, force_weight=args.force_weight
    )
    print("Initial training metrics:", before)
    history = fit_s2ef(
        model,
        loader,
        epochs=args.epochs,
        lr=args.lr,
        force_weight=args.force_weight,
        device=args.device,
        validation=validation,
    )
    after = evaluate_s2ef(
        model, loader, device=args.device, force_weight=args.force_weight
    )
    metadata = {
        "purpose": "Initial development run; not a validated general OC20 predictor",
        "seed": args.seed,
        "training_records": len(dataset),
        "validation_records": len(validation.dataset) if validation is not None else 0,
        "source": str(args.data.resolve()) if args.data else "analytic spring demo",
        "initial_train": before,
        "final_train": after,
        "history": history,
    }
    if validation is not None:
        metadata["final_validation"] = evaluate_s2ef(
            model, validation, device=args.device, force_weight=args.force_weight
        )
    save_shared_checkpoint(model, args.checkpoint, metadata=metadata)
    args.checkpoint.with_suffix(".metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Final training metrics:", after)
    print("Checkpoint:", args.checkpoint.resolve())


if __name__ == "__main__":
    main()
