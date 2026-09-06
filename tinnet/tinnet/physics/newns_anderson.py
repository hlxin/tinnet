"""General Newns-Anderson chemisorption theory module.

The d-band of the surface site is represented by a semi-elliptic density of
states with center :math:`\\epsilon_d` and half width :math:`W_d`.  Each
adsorbate frontier orbital :math:`a` (energy :math:`\\epsilon_a`, coupling
coefficient :math:`\\beta_a`, sp broadening :math:`\\delta_a`) hybridizes with
that band through the chemisorption function

.. math::

    \\Delta_a(\\epsilon) = \\pi \\beta_a V_{ad}^2 \\rho_d(\\epsilon) + \\delta_a,

whose Hilbert transform :math:`\\Lambda_a` enters the hybridization energy

.. math::

    E_{hyb,a} = \\frac{2}{\\pi} \\int_{-\\infty}^{E_F}
        \\arctan \\frac{\\Delta_a}{\\epsilon - \\epsilon_a - \\Lambda_a}\\, d\\epsilon
        - (\\text{same with } V_{ad}^2 = 0).

The adsorption energy is

.. math::

    E_{ad} = E_{sp} + \\sum_a g_a \\left[ E_{hyb,a}
        + 2 (n_a + f)\\, \\alpha_a \\beta_a V_{ad}^2 \\right],

with orbital degeneracy :math:`g_a`, free-adsorbate occupancy :math:`n_a`,
d-band filling :math:`f`, and orthogonalization coefficient :math:`\\alpha_a`.

The whole expression is written once, batched over rows, and is used both for
checkpoint inference and for SHAP explanations.  Nothing in this module is
specific to OH or O: the orbital list comes from an
:class:`~tinnet.tinnet.physics.adsorbates.AdsorbateSpec`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .adsorbates import AdsorbateSpec, adsorbate_spec
from .base import PhysicsModel, register_physics


def hilbert_multiplier(n_grid: int, *, dtype=torch.float32, device=None) -> torch.Tensor:
    """FFT multiplier that turns ``ifft(fft(x) * h)`` into the analytic signal."""
    h = torch.zeros(n_grid, dtype=dtype, device=device)
    if n_grid % 2 == 0:
        h[0] = 1
        h[n_grid // 2] = 1
        h[1:n_grid // 2] = 2
    else:
        h[0] = 1
        h[1:(n_grid + 1) // 2] = 2
    return h


def hilbert_transform(values: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Imaginary part of the analytic signal along the last axis."""
    return torch.imag(torch.fft.ifft(torch.fft.fft(values, dim=-1) * h, dim=-1))


def safe_denominator(value: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace near-zero denominators by ``±eps`` while preserving sign."""
    small = torch.abs(value) <= eps
    return value * (~small) + eps * small * (value >= 0) - eps * small * (value < 0)


def semi_ellipse_dos(ergy: torch.Tensor, d_cen: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """Normalized semi-elliptic d-band DOS for a batch of (center, half width).

    Parameters
    ----------
    ergy : (n_grid,)
    d_cen, width : (B,)

    Returns
    -------
    (B, n_grid) tensor integrating to one on ``ergy``.
    """
    x = (ergy[None, :] - d_cen[:, None]) / width[:, None]
    dos_d = torch.abs(1 - x ** 2) ** 0.5
    dos_d = dos_d * (torch.abs(ergy[None, :] - d_cen[:, None]) < width[:, None])
    area = torch.trapz(dos_d, ergy, dim=1)
    dos_d = dos_d + (area[:, None] <= 1e-10) / len(ergy)
    return dos_d / torch.trapz(dos_d, ergy, dim=1)[:, None]


def wrapped_arctan(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """arctan branch used by the Newns-Anderson phase-shift integral."""
    arctan = torch.atan(numerator / denominator)
    return (arctan - np.pi) * (arctan > 0) + arctan * (arctan <= 0)


@register_physics("newns_anderson")
class NewnsAndersonModel(PhysicsModel):
    """Semi-elliptic Newns-Anderson adsorption-energy theory module.

    Parameters
    ----------
    spec
        An :class:`AdsorbateSpec` or the registered name of one (``'OH'``).
    energy_grid
        ``(e_min, e_max, n_points)`` of the energy grid in eV relative to the
        Fermi level.  The published checkpoints use ``(-15, 15, 3001)``.
    """

    property_label = "$E_{ad}$ (eV)"

    def __init__(self, spec: AdsorbateSpec | str = "OH", energy_grid=(-15.0, 15.0, 3001)):
        if isinstance(spec, str):
            spec = adsorbate_spec(spec)
        self.spec = spec
        self.energy_grid = tuple(energy_grid)
        self.n_latent = spec.n_latent
        self.constant_names = spec.constant_names
        self.parameter_names = spec.parameter_names
        self.parameter_labels = spec.parameter_labels
        self.property_label = spec.property_label
        self._grid_cache: dict = {}

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------
    def _grid(self, dtype, device):
        key = (dtype, str(device))
        if key not in self._grid_cache:
            e_min, e_max, n_points = self.energy_grid
            ergy = torch.linspace(e_min, e_max, n_points, dtype=dtype, device=device)
            h = hilbert_multiplier(n_points, dtype=dtype, device=device)
            fermi = int(torch.argmin(torch.abs(ergy)).item()) + 1
            self._grid_cache[key] = (ergy, h, fermi)
        return self._grid_cache[key]

    # ------------------------------------------------------------------
    # latent -> parameters
    # ------------------------------------------------------------------
    def parameters(self, latent: torch.Tensor, constants: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Apply physical constraints to raw GNN outputs and assemble parameters.

        Latent layout: ``[adse_1, beta_1, delta_1, ..., adse_n, beta_n, delta_n
        (, d_cen, width)]``.  Couplings, broadenings and the band width are made
        positive with a softplus; resonance energies and the band center are
        unconstrained.
        """
        if latent.shape[-1] != self.n_latent:
            raise ValueError(
                f"{self.spec.name}: expected {self.n_latent} latent outputs, got {latent.shape[-1]}."
            )
        batch = latent.shape[0]
        vad2 = torch.as_tensor(constants["vad2"], dtype=latent.dtype, device=latent.device).reshape(batch)

        n_orb = self.spec.n_orbitals
        if self.spec.d_band == "latent":
            d_cen = latent[:, 3 * n_orb]
            width = F.softplus(latent[:, 3 * n_orb + 1])
        else:
            d_cen = torch.as_tensor(constants["d_cen"], dtype=latent.dtype, device=latent.device).reshape(batch)
            width = torch.as_tensor(constants["width"], dtype=latent.dtype, device=latent.device).reshape(batch)

        columns = [vad2, d_cen, width]
        for k in range(n_orb):
            columns.append(latent[:, 3 * k])                 # adsorbate resonance energy
            columns.append(F.softplus(latent[:, 3 * k + 1]))  # coupling coefficient
            columns.append(F.softplus(latent[:, 3 * k + 2]))  # sp broadening
        return torch.stack(columns, dim=1)

    # ------------------------------------------------------------------
    # parameters -> property
    # ------------------------------------------------------------------
    def _orbital_terms(self, *, ergy, h, fermi, vad2, dos_d, adse, beta, delta, eps):
        """Hybridization energy and free-orbital occupancy for one orbital (batched)."""
        wdos = np.pi * (beta[:, None] * vad2[:, None] * dos_d) + delta[:, None]
        wdos_reference = np.pi * (0 * vad2[:, None] * dos_d) + delta[:, None]

        lam = hilbert_transform(wdos, h[None, :])
        arctan = wrapped_arctan(wdos, safe_denominator(ergy[None, :] - adse[:, None] - lam, eps))
        d_hyb = 2 / np.pi * torch.trapz(arctan[:, :fermi], ergy[:fermi], dim=1)

        lorentzian = (1 / np.pi) * delta[:, None] / ((ergy[None, :] - adse[:, None]) ** 2 + delta[:, None] ** 2)
        occupancy = torch.trapz(lorentzian[:, :fermi], ergy[:fermi], dim=1)

        arctan_ref = wrapped_arctan(wdos_reference, safe_denominator(ergy[None, :] - adse[:, None], eps))
        d_hyb_ref = 2 / np.pi * torch.trapz(arctan_ref[:, :fermi], ergy[:fermi], dim=1)
        return d_hyb - d_hyb_ref, occupancy

    def decompose(self, params: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate the theory and return all intermediate physical quantities.

        Returns a dict with ``energy`` (B,), ``filling`` (B,), ``dos_d``
        (B, n_grid), ``ergy`` (n_grid,), and per orbital ``hyb_<name>``,
        ``occupancy_<name>``, ``ortho_<name>``, ``contribution_<name>`` (B,).
        """
        params = torch.as_tensor(params)
        if params.dim() == 1:
            params = params[None, :]
        ergy, h, fermi = self._grid(params.dtype, params.device)
        eps = np.finfo(float).eps

        vad2, d_cen, width = params[:, 0], params[:, 1], params[:, 2]
        dos_d = semi_ellipse_dos(ergy, d_cen, width)
        filling = torch.trapz(dos_d[:, :fermi], ergy[:fermi], dim=1)

        out: dict[str, torch.Tensor] = {"ergy": ergy, "dos_d": dos_d, "filling": filling}
        energy = torch.full_like(vad2, float(self.spec.esp))
        for k, orbital in enumerate(self.spec.orbitals):
            adse = params[:, 3 + 3 * k]
            beta = params[:, 4 + 3 * k]
            delta = params[:, 5 + 3 * k]
            hyb, occupancy = self._orbital_terms(
                ergy=ergy, h=h, fermi=fermi, vad2=vad2, dos_d=dos_d,
                adse=adse, beta=beta, delta=delta, eps=eps,
            )
            ortho = 2 * (occupancy + filling) * orbital.alpha * beta * vad2
            contribution = orbital.degeneracy * (hyb + ortho)
            energy = energy + contribution
            out[f"hyb_{orbital.name}"] = hyb
            out[f"occupancy_{orbital.name}"] = occupancy
            out[f"ortho_{orbital.name}"] = ortho
            out[f"contribution_{orbital.name}"] = contribution
        out["energy"] = energy
        return out

    def property_from_parameters(self, params: torch.Tensor) -> torch.Tensor:
        return self.decompose(params)["energy"]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def projected_dos(self, params, orbital: str | None = None):
        """Return ``(ergy, dos)`` NumPy arrays for plotting.

        Without ``orbital`` the semi-elliptic d-DOS is returned.  With an
        orbital name the adsorbate-projected DOS
        :math:`\\Delta / ((\\epsilon-\\epsilon_a-\\Lambda)^2+\\Delta^2) / \\pi` is
        returned instead.
        """
        with torch.no_grad():
            params = torch.as_tensor(np.asarray(params, dtype=np.float32))
            if params.dim() == 1:
                params = params[None, :]
            ergy, h, _ = self._grid(params.dtype, params.device)
            vad2, d_cen, width = params[:, 0], params[:, 1], params[:, 2]
            dos_d = semi_ellipse_dos(ergy, d_cen, width)
            if orbital is None:
                return ergy.numpy(), dos_d.numpy()
            k = [o.name for o in self.spec.orbitals].index(orbital)
            adse, beta, delta = params[:, 3 + 3 * k], params[:, 4 + 3 * k], params[:, 5 + 3 * k]
            wdos = np.pi * (beta[:, None] * vad2[:, None] * dos_d) + delta[:, None]
            lam = hilbert_transform(wdos, h[None, :])
            dos_a = wdos / ((ergy[None, :] - adse[:, None] - lam) ** 2 + wdos ** 2) / np.pi
            return ergy.numpy(), dos_a.numpy()
