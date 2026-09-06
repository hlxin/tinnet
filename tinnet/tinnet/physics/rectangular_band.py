"""Rectangular d-band theory module (filling, center, width of a site).

The second moment of the site d band is built from tight-binding couplings to
its neighbors :math:`j` at distance :math:`d_{ij}`,

.. math::

    V^2_{ds,j} \\propto r_{d,i}^{3} / d_{ij}^{7}, \\qquad
    V^2_{dd,j} \\propto r_{d,i}^{3} r_{d,j}^{3} / d_{ij}^{10},

each renormalized by a learned bond relaxation factor :math:`\\zeta_j`:

.. math::

    m_2 = \\sum_j \\left( V^2_{ds,j} / \\zeta_j^{7} + V^2_{dd,j} / \\zeta_j^{10} \\right).

The rectangular band of width :math:`W = \\sqrt{12 m_2}` has its center at

.. math::

    \\epsilon_d = \\beta \\sqrt{m_2}
        \\left( \\epsilon_d^{bulk} / W^{bulk} - \\alpha\\, \\Delta\\chi \\right),

with a learned resonance scale :math:`\\beta`, a learned charge-transfer
coefficient :math:`\\alpha`, and the Mulliken electronegativity difference
:math:`\\Delta\\chi` to the first neighbor shell.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .base import PhysicsModel, register_physics

# Harrison solid-state-table prefactors used by the published TinNet checkpoints.
V2DS_PREFACTOR = 9.9856 * 7.62 ** 2
V2DD_PREFACTOR = 415.565 * 7.62 ** 2


def hopping_couplings(site_radius, neighbor_radii, distances):
    """Squared d-s and d-d hopping integrals for a site and its neighbors.

    Works for NumPy arrays and torch tensors; ``neighbor_radii``/``distances``
    may carry a leading batch dimension in which case ``site_radius`` is
    broadcast as ``(B, 1)``.
    """
    vds = site_radius ** 1.5 / distances ** 3.5
    vdd = site_radius ** 1.5 * neighbor_radii ** 1.5 / distances ** 5.0
    return V2DS_PREFACTOR * vds ** 2.0, V2DD_PREFACTOR * vdd ** 2.0


@register_physics("rectangular_band")
class RectangularBandModel(PhysicsModel):
    """Rectangular d-band theory module for one surface/bulk site.

    Parameters
    ----------
    max_neighbors
        Padded neighbor count ``M`` (86 for the published checkpoints).

    Latent layout (``M + 3``): ``[zeta_raw (M), alpha_raw, beta_raw, filling_raw]``.

    Physical parameter layout (``4M + 6``), which is also the SHAP feature vector::

        0          site d-orbital radius
        1 : 1+M    neighbor d-orbital radii             -> ligand effect
        1+M : 1+2M neighbor distances                   -> strain effect
        1+2M       tabulated bulk d-band center
        2+2M       tabulated bulk full width
        3+2M       Mulliken electronegativity difference -> charge transfer
        4+2M : 4+3M padding filter (1 for real neighbors)
        4+3M : 4+4M zeta^(1/7) bond relaxation           -> local relaxation
        4+4M       alpha charge-transfer coefficient     -> charge transfer
        5+4M       beta resonance scale                  -> resonance
    """

    property_label = r"$d\rm{-band\ center}$ (eV)"
    constant_names = (
        "site_radius", "neighbor_radii", "distances",
        "bulk_center", "bulk_width", "mulliken", "padding_filter",
    )
    effect_labels = {
        "resonance": r"$(\alpha, \xi)$",
        "strain": r"$d_{ij}$",
        "relaxation": r"$\zeta$",
        "ligand": r"$(\lambda, r_{dj})$",
        "charge_transfer": r"$(\beta, \Delta\chi)$",
    }

    def __init__(self, max_neighbors: int = 86):
        M = self.max_neighbors = int(max_neighbors)
        self.n_latent = M + 3
        self.slices = {
            "site_radius": 0,
            "neighbor_radii": slice(1, 1 + M),
            "distances": slice(1 + M, 1 + 2 * M),
            "bulk_center": 1 + 2 * M,
            "bulk_width": 2 + 2 * M,
            "mulliken": 3 + 2 * M,
            "padding_filter": slice(4 + 2 * M, 4 + 3 * M),
            "zeta": slice(4 + 3 * M, 4 + 4 * M),
            "alpha": 4 + 4 * M,
            "beta": 5 + 4 * M,
        }
        names = ["site_radius"]
        names += [f"neighbor_radius_{j}" for j in range(M)]
        names += [f"distance_{j}" for j in range(M)]
        names += ["bulk_center", "bulk_width", "mulliken"]
        names += [f"padding_{j}" for j in range(M)]
        names += [f"zeta_{j}" for j in range(M)]
        names += ["alpha", "beta"]
        self.parameter_names = tuple(names)
        self.parameter_labels = tuple(names)
        # SHAP columns aggregated into the five physical effects.
        idx = np.arange(len(names))
        self.effect_columns = {
            "resonance": idx[[self.slices["beta"]]],
            "strain": idx[self.slices["distances"]],
            "relaxation": idx[self.slices["zeta"]],
            "ligand": idx[self.slices["neighbor_radii"]],
            "charge_transfer": np.concatenate((idx[[self.slices["alpha"]]], idx[[self.slices["mulliken"]]])),
        }

    # ------------------------------------------------------------------
    # latent -> state variables
    # ------------------------------------------------------------------
    def state_variables(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Constrained neural state variables from the raw GNN outputs."""
        M = self.max_neighbors
        return dict(
            zeta=F.softplus(latent[:, :M]),          # bond relaxation, > 0
            alpha=latent[:, M],                      # charge transfer, unconstrained
            beta=F.softplus(latent[:, M + 1]),       # resonance scale, > 0
            filling=torch.sigmoid(latent[:, M + 2]),  # d-band filling in (0, 1)
        )

    @staticmethod
    def second_moment(zeta, v2ds, v2dd):
        """:math:`m_2 = \\sum_j V^2_{ds,j}/\\zeta_j + V^2_{dd,j}/\\zeta_j^{10/7}`."""
        return torch.sum(v2ds / zeta + v2dd / zeta ** (10.0 / 7.0), dim=-1)

    @staticmethod
    def band_center(m2, beta, alpha, bulk_center, bulk_width, mulliken):
        return beta * m2 ** 0.5 * (bulk_center / bulk_width - alpha * mulliken)

    def band_properties(self, latent: torch.Tensor, constants: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Inference path using tabulated couplings ``v2ds``/``v2dd``.

        Returns ``filling``, ``d_cen``, ``full_width`` plus the state variables.
        This is the fast path used with pretrained checkpoints; it is
        algebraically identical to ``property_from_parameters`` when the
        tabulated couplings were produced by :func:`hopping_couplings`.
        """
        s = self.state_variables(latent)
        m2 = self.second_moment(s["zeta"], constants["v2ds"], constants["v2dd"])
        d_cen = self.band_center(m2, s["beta"], s["alpha"],
                                 constants["bulk_center"], constants["bulk_width"], constants["mulliken"])
        s.update(m2=m2, d_cen=d_cen, full_width=(12 * m2) ** 0.5)
        return s

    # ------------------------------------------------------------------
    # PhysicsModel contract
    # ------------------------------------------------------------------
    def parameters(self, latent: torch.Tensor, constants: Mapping[str, torch.Tensor]) -> torch.Tensor:
        batch = latent.shape[0]
        s = self.state_variables(latent)

        def const(name, k=None):
            value = torch.as_tensor(constants[name], dtype=latent.dtype, device=latent.device)
            return value.reshape(batch, k) if k else value.reshape(batch, 1)

        return torch.cat((
            const("site_radius"),
            const("neighbor_radii", self.max_neighbors),
            const("distances", self.max_neighbors),
            const("bulk_center"), const("bulk_width"), const("mulliken"),
            const("padding_filter", self.max_neighbors),
            s["zeta"] ** (1.0 / 7.0),
            s["alpha"][:, None],
            s["beta"][:, None],
        ), dim=1)

    def property_from_parameters(self, params: torch.Tensor) -> torch.Tensor:
        sl = self.slices
        site_radius = params[:, sl["site_radius"]][:, None]
        v2ds, v2dd = hopping_couplings(site_radius, params[:, sl["neighbor_radii"]], params[:, sl["distances"]])
        padding = params[:, sl["padding_filter"]]
        zeta7 = params[:, sl["zeta"]]
        m2 = torch.sum((v2ds * padding) / zeta7 ** 7.0 + (v2dd * padding) / zeta7 ** 10.0, dim=1)
        return self.band_center(m2, params[:, sl["beta"]], params[:, sl["alpha"]],
                                params[:, sl["bulk_center"]], params[:, sl["bulk_width"]], params[:, sl["mulliken"]])

    def shap_function(self, params) -> np.ndarray:
        """Double-precision NumPy evaluation (the SHAP inputs are float64)."""
        with torch.no_grad():
            params = torch.as_tensor(np.atleast_2d(np.asarray(params, dtype=np.float64)))
            return np.atleast_1d(self.property_from_parameters(params).numpy())

    def aggregate_effects(self, shap_values) -> np.ndarray:
        """Sum raw per-feature SHAP values into the five physical effects."""
        raw = np.asarray(shap_values)
        return np.array([raw[:, cols].sum() for cols in self.effect_columns.values()])

    @property
    def effect_names(self) -> tuple[str, ...]:
        return tuple(self.effect_columns)
