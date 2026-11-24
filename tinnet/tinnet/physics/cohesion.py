"""Renormalized-atom cohesion theory module (cohesive energy).

Per atom the cohesive energy is decomposed as

.. math::

    E_{coh} = E_{prom} + E_{renorm} + E_{s} + E_{d},

with the tabulated promotion energy :math:`E_{prom}`, a learned
renormalization term, the free-electron-like conduction-band term
:math:`E_s = c\\,[n_s / (\\beta V_{ws})]^{2/3}` and the rectangular-band d
term :math:`E_d = -\\alpha\\,W/20\\,n_d (10 - n_d)`.

:math:`c = (3\\hbar^2/10m)(3\\pi^2)^{2/3} = 21.88` eV Angstrom^2, so the
Wigner-Seitz volumes must be given in Angstrom^3 (the unit of the tabulated
``wigner_seitz_volume`` constants).  :math:`\\beta` rescales the tabulated
volume to an effective local volume and :math:`\\alpha` corrects the
rectangular-band estimate.  This is the parameterization the pretrained
checkpoints in ``data/pretrained/cohesive_energy`` (April 2026) were trained
with.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .base import PhysicsModel, register_physics

# (3 hbar^2 / 10 m) (3 pi^2)^(2/3) in eV * Angstrom^2.  The value used before,
# 2.188e-19, is the same constant in eV * m^2; combined with volumes in
# Angstrom^3 it made the conduction-band term ~1e-20 eV.
FREE_ELECTRON_PREFACTOR = 21.880420859580444


@register_physics("cohesion")
class CohesionModel(PhysicsModel):
    """Cohesive-energy theory module evaluated per atom.

    Latent layout (6 per atom): ``[e_ren, alpha, beta, n_s, n_d, width]`` where
    all but ``e_ren`` are made positive by a softplus.  ``alpha`` multiplies the
    rectangular-band d term and ``beta`` scales the Wigner-Seitz volume in the
    conduction-band term.
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

        conduction_energy = FREE_ELECTRON_PREFACTOR * (s["ns"] / (s["beta"] * vws)) ** (2.0 / 3.0)
        d_band_energy = s["alpha"] * s["width"] / 20.0 * s["nd"] * (s["nd"] - 10.0)
        return torch.stack((pe, s["e_ren"], conduction_energy, d_band_energy), dim=1)

    def property_from_parameters(self, params: torch.Tensor) -> torch.Tensor:
        return params.sum(dim=-1)
