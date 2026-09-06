"""Energy/force training, evaluation, and portable shared-model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from .shared_adsorption import SharedAdsorptionModel
from .structures import move_batch


def s2ef_loss(model, batch, *, energy_weight=1.0, force_weight=10.0, create_graph=True):
    """MSE on energies and free-atom force components, with count-aware metrics.

    Positive-weight terms require labels. Fixed atoms remain in the graph but
    their force labels are masked. Force training uses second derivatives.
    """
    if energy_weight < 0 or force_weight < 0 or energy_weight + force_weight <= 0:
        raise ValueError(
            "Use nonnegative loss weights with at least one positive term."
        )
    if batch.get("energy_reference") != model.energy_reference:
        raise ValueError("Batch and model energy reference conventions must match.")
    for key, weight in (("energy", energy_weight), ("forces", force_weight)):
        if weight > 0 and key not in batch:
            raise ValueError(f"Missing {key} labels for the requested loss.")
    output = model(
        batch["graph"],
        compute_forces=force_weight > 0,
        create_graph=create_graph and force_weight > 0,
    )
    loss = output["energy"].sum() * 0.0
    metrics = {}
    if energy_weight:
        target = batch["energy"].to(output["energy"])
        loss = loss + energy_weight * F.mse_loss(output["energy"], target)
        metrics.update(
            energy_abs=float((output["energy"] - target).detach().abs().sum()),
            n_energy=target.numel(),
        )
    if force_weight:
        free = ~batch["graph"]["fixed_mask"]
        if not free.any():
            raise ValueError("Force training requires at least one free atom.")
        predicted = output["forces"][free]
        target = batch["forces"][free].to(predicted)
        loss = loss + force_weight * F.mse_loss(predicted, target)
        metrics.update(
            force_abs=float((predicted - target).detach().abs().sum()),
            n_force=target.numel(),
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite energy/force loss.")
    return loss, metrics


def evaluate_s2ef(
    model, batches, *, device="cpu", energy_weight=1.0, force_weight=10.0
):
    """Energy MAE (eV) and free force-component MAE (eV/Angstrom).

    Autograd computes forces without second-derivative graphs. Metrics are
    weighted by configurations/components, including uneven final batches.
    """
    model.to(device).eval()
    totals = {"energy_abs": 0.0, "n_energy": 0, "force_abs": 0.0, "n_force": 0}
    n_batches = 0
    for batch in batches:
        _, metrics = s2ef_loss(
            model,
            move_batch(batch, device),
            energy_weight=energy_weight,
            force_weight=force_weight,
            create_graph=False,
        )
        for key, value in metrics.items():
            totals[key] += value
        n_batches += 1
    if not n_batches:
        raise ValueError("Evaluation loader is empty.")
    return {
        name + "_mae": totals[name + "_abs"] / totals["n_" + name]
        for name in ("energy", "force")
        if totals["n_" + name]
    }


def fit_s2ef(
    model,
    batches,
    *,
    epochs=10,
    lr=1e-3,
    energy_weight=1.0,
    force_weight=10.0,
    device="cpu",
    validation=None,
    log_every=1,
):
    """Train on a re-iterable loader/list. Split validation by system, not frame."""
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs and lr must be positive.")
    if iter(batches) is batches:
        raise ValueError("batches must be a re-iterable loader or list.")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = s2ef_loss(
                model,
                move_batch(batch, device),
                energy_weight=energy_weight,
                force_weight=force_weight,
                create_graph=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.detach())
            n += 1
        if not n:
            raise ValueError("Training loader is empty.")
        row = {"epoch": epoch + 1, "train_loss": total_loss / n}
        if validation is not None:
            row.update(
                {
                    "val_" + k: v
                    for k, v in evaluate_s2ef(
                        model,
                        validation,
                        device=device,
                        energy_weight=energy_weight,
                        force_weight=force_weight,
                    ).items()
                }
            )
        history.append(row)
        if log_every and (epoch + 1) % log_every == 0:
            print(row)
    return history


def save_shared_checkpoint(model, path, *, metadata=None):
    """Store architecture, weights, precision, and energy convention together."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "tinnet-shared-s2ef-v1",
            "config": model.config,
            "state_dict": model.state_dict(),
            "dtype": str(next(model.parameters()).dtype).split(".")[-1],
            "metadata": metadata or {},
        },
        path,
    )
    return path


def load_shared_checkpoint(path, *, device="cpu"):
    """Reconstruct an S2EF model; legacy OH/O checkpoints are incompatible."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "tinnet-shared-s2ef-v1":
        raise ValueError(
            "Expected a shared S2EF checkpoint, not a legacy TinNet checkpoint."
        )
    dtype = checkpoint.get("dtype", "float32")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"Unsupported checkpoint precision: {dtype}")
    model = SharedAdsorptionModel(**checkpoint["config"]).to(
        device=device, dtype=getattr(torch, dtype)
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
