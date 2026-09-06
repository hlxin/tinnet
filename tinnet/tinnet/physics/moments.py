"""Tight-binding moment theory module (second, third, fourth d-band moments).

Given the tight-binding hopping matrix :math:`H` of a site and its neighbors
(orbitals ``s, dxy, dyz, dxz, dz2, dx2-y2``), the :math:`n`-th moment of the
site-projected d DOS is :math:`\\mu_n = \\mathrm{Tr}_d\\,[H^n]_{00}`.

The GNN predicts three correction fields per atom pair, ``(gamma_ds,
gamma_dd, zeta)``, that renormalize the Harrison hopping integrals:

.. math::

    H_{corr} = H \\odot \\left( \\zeta^{2/3.5} P_{ss} + \\zeta P_{ds}
        + \\zeta^{5/3.5} P_{dd} + \\gamma_{ds} P^{\\Gamma}_{ds}
        + \\gamma_{dd} P^{\\Gamma}_{dd} \\right),

where the :math:`P` tensors are 0/1 masks selecting hopping channels.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .base import PhysicsModel, register_physics


@register_physics("moments")
class MomentModel(PhysicsModel):
    """Moment theory module.

    Here the "physical parameters" are the corrected hopping tensor of shape
    ``(B, P, 6, P, 6)`` rather than a flat vector, so ``parameters`` and
    ``property_from_parameters`` operate on that tensor.  ``n_latent`` is the
    number of correction channels per atom pair.
    """

    n_latent = 3
    constant_names = ("hopping", "power_ss", "power_ds", "power_dd", "power_gamma_ds", "power_gamma_dd")
    parameter_names = ("hopping_matrix",)
    parameter_labels = (r"$H_{corr}$",)
    property_label = r"$\mu_2$ (eV$^2$)"
    moment_names = ("m2", "m3", "m4")

    @staticmethod
    def apply_corrections(raw_corrections: torch.Tensor, constants: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Convert raw pairwise GNN outputs ``(B, P, P, 3)`` into a corrected hopping tensor."""
        zeta = F.softplus(raw_corrections[..., 2])[:, :, None, :, None]
        gamma_ds = raw_corrections[..., 0][:, :, None, :, None]
        gamma_dd = raw_corrections[..., 1][:, :, None, :, None]
        corrected_power = (
            torch.pow(zeta, 2.0 / 3.5) * constants["power_ss"]
            + zeta * constants["power_ds"]
            + torch.pow(zeta, 5.0 / 3.5) * constants["power_dd"]
            + gamma_ds * constants["power_gamma_ds"]
            + gamma_dd * constants["power_gamma_dd"]
        )
        return constants["hopping"] * corrected_power

    def parameters(self, latent: torch.Tensor, constants: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.apply_corrections(latent, constants)

    @staticmethod
    def moments(hopping: torch.Tensor) -> torch.Tensor:
        """Second, third and fourth site-projected d moments, shape ``(B, 3)``."""
        hh2 = [torch.tensordot(h1, h1, ([2, 3], [0, 1])) for h1 in hopping]
        hh3 = [torch.tensordot(h2, h1, ([2, 3], [0, 1])) for h2, h1 in zip(hh2, hopping)]
        hh4 = [torch.tensordot(h3, h1, ([2, 3], [0, 1])) for h3, h1 in zip(hh3, hopping)]
        m2 = torch.stack([torch.sum(torch.diag(h[0, 1:, 0, 1:])) for h in hh2])
        m3 = torch.stack([torch.sum(torch.diag(h[0, 1:, 0, 1:])) for h in hh3])
        m4 = torch.stack([torch.sum(torch.diag(h[0, 1:, 0, 1:])) for h in hh4])
        return torch.stack((m2, m3, m4)).T

    def property_from_parameters(self, params: torch.Tensor) -> torch.Tensor:
        return self.moments(params)[:, 0]

    def forward(self, latent, constants=None):
        constants = {} if constants is None else constants
        self.check_constants(constants)
        hopping = self.parameters(latent, constants)
        return self.moments(hopping), hopping
