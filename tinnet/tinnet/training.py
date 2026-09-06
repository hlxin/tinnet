"""Theory-infused training scaffold shared by all TinNet properties.

A :class:`TheoryInfusedModel` is simply ``physics(gnn(graph), constants)``.
Because every :class:`~tinnet.tinnet.physics.base.PhysicsModel` is
differentiable, the loss on the target property back-propagates through the
theory equation into the GNN, and optional auxiliary losses can be placed on
any physical parameter (for example a DFT d-band center) to regularize the
latent space.  New properties or adsorbates therefore train with exactly the
same loop as the published models.

Example
-------
>>> physics = NewnsAndersonModel(register_adsorbate(my_spec))
>>> gnn = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, n_out=physics.n_latent, readout='site')
>>> model = TheoryInfusedModel(gnn, physics)
>>> history = fit(model, batches, epochs=200, lr=1e-3,
...               parameter_weights={'d_cen': 0.1})   # optional auxiliary target
>>> save_checkpoint(model, 'data/pretrained/adsorption_energy/N/atop/model_0.pth.tar')
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn

from .physics.base import PhysicsModel

Batch = Mapping[str, object]


class TheoryInfusedModel(nn.Module):
    """GNN encoder followed by a physics head.

    ``forward(inputs, constants)`` returns ``(property, parameters)`` where
    ``inputs`` is the tuple passed to ``gnn(*inputs)`` and ``constants`` is the
    mapping of tabulated constants required by ``physics.constant_names``.
    """

    def __init__(self, gnn: nn.Module, physics: PhysicsModel):
        super().__init__()
        self.gnn = gnn
        self.physics = physics

    def forward(self, inputs, constants=None):
        latent = self.gnn(*inputs)
        return self.physics.forward(latent, constants)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    if isinstance(value, Mapping):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def theory_loss(model: TheoryInfusedModel, batch: Batch, *, parameter_weights: Mapping[str, float] | None = None,
                criterion: Callable = nn.functional.mse_loss) -> tuple[torch.Tensor, dict[str, float]]:
    """Property loss plus weighted auxiliary losses on physical parameters.

    ``batch`` keys: ``inputs`` (tuple for the GNN), ``constants`` (mapping),
    ``target`` (B,), and optionally ``parameter_targets`` mapping a parameter
    name from ``physics.parameter_names`` to a (B,) tensor.
    """
    prediction, params = model(batch["inputs"], batch.get("constants"))
    target = torch.as_tensor(batch["target"], dtype=prediction.dtype, device=prediction.device).reshape(prediction.shape)
    loss = criterion(prediction, target)
    parts = {"property": float(loss.detach())}
    for name, weight in (parameter_weights or {}).items():
        value = batch.get("parameter_targets", {}).get(name)
        if value is None:
            continue
        column = params[:, model.physics.parameter_index(name)]
        value = torch.as_tensor(value, dtype=column.dtype, device=column.device).reshape(column.shape)
        aux = criterion(column, value)
        parts[name] = float(aux.detach())
        loss = loss + weight * aux
    return loss, parts


def fit(model: TheoryInfusedModel, batches: Iterable[Batch] | Callable[[], Iterable[Batch]], *,
        epochs: int = 100, lr: float = 1e-3, weight_decay: float = 0.0,
        parameter_weights: Mapping[str, float] | None = None, device=None,
        optimizer: torch.optim.Optimizer | None = None, log_every: int = 0,
        validation: Iterable[Batch] | Callable[[], Iterable[Batch]] | None = None) -> dict[str, list[float]]:
    """Train a theory-infused model and return the loss history.

    ``batches`` may be a list of batches or a zero-argument callable returning
    a fresh iterable each epoch (for example a ``DataLoader``).
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, list[float]] = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for batch in (batches() if callable(batches) else batches):
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            loss, _ = theory_loss(model, batch, parameter_weights=parameter_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            n += 1
        history["train"].append(epoch_loss / max(n, 1))

        if validation is not None:
            history["val"].append(evaluate(model, validation, parameter_weights=parameter_weights, device=device))
        if log_every and (epoch + 1) % log_every == 0:
            msg = f"epoch {epoch + 1:5d}  train {history['train'][-1]:.4e}"
            if validation is not None:
                msg += f"  val {history['val'][-1]:.4e}"
            print(msg)
    return history


@torch.no_grad()
def evaluate(model: TheoryInfusedModel, batches, *, parameter_weights=None, device=None) -> float:
    """Mean loss over ``batches``."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    for batch in (batches() if callable(batches) else batches):
        loss, _ = theory_loss(model, _to_device(batch, device), parameter_weights=parameter_weights)
        total += float(loss)
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def predict(model: TheoryInfusedModel, batch: Batch, device=None):
    """Return ``(property, parameters)`` as NumPy arrays."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    batch = _to_device(batch, device)
    prediction, params = model(batch["inputs"], batch.get("constants"))
    return prediction.cpu().numpy(), params.cpu().numpy()


def save_checkpoint(model: TheoryInfusedModel, path: str | Path, **extra) -> Path:
    """Save the GNN weights in the ``{'state_dict': ...}`` layout used by TinNet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.gnn.state_dict(), "physics": model.physics.name, **extra}, path)
    return path


def ensemble_predict(models: Iterable[TheoryInfusedModel], batch: Batch, device=None):
    """Stack predictions of several ensemble members: ``(n_models, B)``, ``(n_models, B, n_params)``."""
    outputs = [predict(m, batch, device=device) for m in models]
    return np.stack([o[0] for o in outputs]), np.stack([o[1] for o in outputs])
