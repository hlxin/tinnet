"""Base contract for TinNet theory ("physics") modules.

A TinNet model is a graph neural network (GNN) followed by a *physics head*.
The GNN produces a small vector of raw latent outputs; the physics head turns
those outputs, together with tabulated constants of the structure, into
physically meaningful parameters and finally into the target property through
a closed-form theory expression.

Every theory module in TinNet implements the same three-step contract::

    latent (from GNN)  --parameters()-->  physical parameters
    physical parameters --property_from_parameters()-->  property

Because ``property_from_parameters`` is *the* single implementation of the
theory equation, the same object is used for

* checkpoint inference (``forward``),
* SHAP explanations on the physical parameters (``shap_function``), and
* training, where gradients flow from the property back through the theory
  equation into the GNN (``forward`` is differentiable).

To add a new property, subclass :class:`PhysicsModel`, implement the two
abstract methods, declare ``n_latent`` / ``parameter_names`` /
``constant_names`` and register the class with :func:`register_physics`.
"""

from __future__ import annotations

import abc
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

TensorLike = torch.Tensor | np.ndarray | Sequence[float]


def _as_batched_tensor(values, *, dtype=torch.float32, device=None, ndim=1) -> torch.Tensor:
    """Convert ``values`` to a tensor with at least ``ndim`` leading batch dims.

    Existing tensors keep their dtype (float64 stays float64); other inputs
    are converted to ``dtype``.
    """
    if torch.is_tensor(values):
        tensor = values if device is None else values.to(device)
    else:
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
    while tensor.dim() < ndim:
        tensor = tensor.unsqueeze(0)
    return tensor


class PhysicsModel(abc.ABC):
    """Abstract theory module: GNN latent outputs -> parameters -> property.

    Attributes that subclasses must define
    --------------------------------------
    name : str
        Registry name of the theory module.
    n_latent : int
        Number of raw GNN outputs consumed *per row* (per site, per atom, or per
        bond, depending on the GNN readout).
    parameter_names : tuple[str, ...]
        Names of the physical parameters, in the column order produced by
        :meth:`parameters`.  These are the SHAP features.
    parameter_labels : tuple[str, ...]
        Plot labels (LaTeX allowed) for ``parameter_names``.
    constant_names : tuple[str, ...]
        Tabulated constants that must be supplied for each row (for example
        ``vad2`` of the adsorption site).  Each is a tensor of shape ``(B,)``
        or ``(B, k)``.
    property_label : str
        Axis label for the predicted property.
    """

    name: str = "physics"
    n_latent: int = 0
    parameter_names: tuple[str, ...] = ()
    parameter_labels: tuple[str, ...] = ()
    constant_names: tuple[str, ...] = ()
    property_label: str = "property"

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def parameters(self, latent: torch.Tensor, constants: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Map raw GNN outputs and tabulated constants to physical parameters.

        Parameters
        ----------
        latent : Tensor, shape (B, n_latent)
            Raw GNN outputs.  Any activation needed to enforce physical
            constraints (softplus for widths, sigmoid for fillings, ...) is
            applied here, so the GNN itself stays theory-agnostic.
        constants : mapping name -> Tensor, shape (B,) or (B, k)
            Tabulated constants listed in ``constant_names``.

        Returns
        -------
        Tensor, shape (B, n_parameters)
        """

    @abc.abstractmethod
    def property_from_parameters(self, params: torch.Tensor) -> torch.Tensor:
        """Evaluate the theory equation on physical parameters.

        This is the only place where the physics equation lives.  It must be
        batched and differentiable with respect to ``params``.

        Parameters
        ----------
        params : Tensor, shape (B, n_parameters)

        Returns
        -------
        Tensor, shape (B,)
        """

    # ------------------------------------------------------------------
    # Derived helpers shared by all theory modules
    # ------------------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    def forward(self, latent: torch.Tensor, constants: Mapping[str, torch.Tensor] | None = None):
        """Full theory-infused evaluation: ``(property, parameters)``."""
        constants = {} if constants is None else constants
        latent = _as_batched_tensor(latent, ndim=2)
        self.check_constants(constants)
        params = self.parameters(latent, constants)
        return self.property_from_parameters(params), params

    __call__ = forward

    def check_constants(self, constants: Mapping[str, torch.Tensor]) -> None:
        """Raise a clear error when a required tabulated constant is missing."""
        missing = [name for name in self.constant_names if name not in constants]
        if missing:
            raise KeyError(
                f"{type(self).__name__} requires constants {list(self.constant_names)}; "
                f"missing: {missing}"
            )

    def shap_function(self, params: TensorLike) -> np.ndarray:
        """NumPy-in / NumPy-out wrapper of the theory equation for ``shap``.

        ``shap.Explainer`` calls the model with a 2D NumPy array of feature
        rows.  The rows are the physical parameters, so the explanation is
        expressed directly in terms of theory quantities.
        """
        with torch.no_grad():
            params = _as_batched_tensor(np.asarray(params, dtype=np.float32), ndim=2)
            return np.atleast_1d(self.property_from_parameters(params).detach().cpu().numpy())

    def parameter_index(self, name: str) -> int:
        """Column index of a physical parameter."""
        try:
            return self.parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown parameter '{name}'. Known: {self.parameter_names}") from exc

    def describe(self) -> str:
        """Human-readable summary of the theory module contract."""
        lines = [
            f"{type(self).__name__} ('{self.name}')",
            f"  latent outputs per row : {self.n_latent}",
            f"  constants required     : {list(self.constant_names)}",
            f"  physical parameters    : {list(self.parameter_names)}",
            f"  property               : {self.property_label}",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
PHYSICS_REGISTRY: dict[str, type[PhysicsModel]] = {}


def register_physics(name: str | None = None) -> Callable[[type[PhysicsModel]], type[PhysicsModel]]:
    """Class decorator registering a theory module under ``name``."""

    def decorator(cls: type[PhysicsModel]) -> type[PhysicsModel]:
        key = name or cls.name
        if key in PHYSICS_REGISTRY and PHYSICS_REGISTRY[key] is not cls:
            raise ValueError(f"A physics model named '{key}' is already registered.")
        cls.name = key
        PHYSICS_REGISTRY[key] = cls
        return cls

    return decorator


def available_physics() -> list[str]:
    """Names of all registered theory modules."""
    return sorted(PHYSICS_REGISTRY)


def get_physics(name: str, **kwargs) -> PhysicsModel:
    """Instantiate a registered theory module by name."""
    if name not in PHYSICS_REGISTRY:
        raise KeyError(f"Unknown physics model '{name}'. Available: {available_physics()}")
    return PHYSICS_REGISTRY[name](**kwargs)
