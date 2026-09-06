"""Renormalized-atom cohesion theory module (cohesive energy).

Per atom the cohesive energy is decomposed as

.. math::

    E_{coh} = E_{prom} + E_{renorm} + E_{s} + E_{d},

with the tabulated promotion energy :math:`E_{prom}`, a learned
renormalization term, a free-electron-like sp term
:math:`E_s = c\\,V_{ws}^{-2/3}\\,\\alpha\\,n_s^{2/3}` and the rectangular-band d
term :math:`E_d = \\beta\\,W/20\\,n_d (n_d - 10)`.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .base import PhysicsModel, register_physics

FREE_ELECTRON_PREFACTOR = 2.1880420859580444e-19


@register_physics("cohesion")
class CohesionModel(PhysicsModel):
    """Cohesive-energy theory module evaluated per atom.

    Latent layout (6 per atom): ``[e_ren, alpha, beta, n_s, n_d, width]`` where
    all but ``e_ren`` are made positive by a softplus.
    """

    n_latent = 6
    constant_names = ("promotion_energy", "wigner_seitz_volume")
    parameter_names = ("E_prom", "E_renorm", "E_s", "E_d")
    parameter_labels = (r"$E_{prom}$", r"$E_{renorm}$", r"$E_{s}$", r"$E_{d}$")
    property_label = r"$E_{Cohesive}$ (eV/atom)"

    def state_variables(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the constrained neural state variables."""
        e_ren = latent[:, 0]
        alpha, beta, ns, nd, width = [F.softplus(latent[:, i]) for i in range(1, 6)]
        return dict(e_ren=e_ren, alpha=alpha, beta=beta, ns=ns, nd=nd, width=width)

    def parameters(self, latent: torch.Tensor, constants: Mapping[str, torch.Tensor]) -> torch.Tensor:
        batch = latent.shape[0]
        pe = torch.as_tensor(constants["promotion_energy"], dtype=latent.dtype, device=latent.device).reshape(batch)
        vws = torch.as_tensor(constants["wigner_seitz_volume"], dtype=latent.dtype, device=latent.device).reshape(batch)
        s = self.state_variables(latent)

        conduction_term = FREE_ELECTRON_PREFACTOR * (1.0 / vws) ** (2.0 / 3.0)
        conduction_energy = conduction_term * s["alpha"] * s["ns"] ** (2.0 / 3.0)
        d_band_energy = s["beta"] * s["width"] / 20.0 * s["nd"] * (s["nd"] - 10.0)
        return torch.stack((pe, s["e_ren"], conduction_energy, d_band_energy), dim=1)

    def property_from_parameters(self, params: torch.Tensor) -> torch.Tensor:
        return params.sum(dim=-1)
