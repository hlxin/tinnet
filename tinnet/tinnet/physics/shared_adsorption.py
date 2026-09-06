"""Smooth independent-channel Newns-Anderson head for force training.

Channels are learned effective states per adsorbate atom, not identified
molecular orbitals. This first approximation neglects off-diagonal orbital
mixing. A smoothed semi-ellipse avoids singular derivatives at band edges;
atan2 implements the phase branch without division by a near-zero value.
The original published-checkpoint head remains unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import PhysicsModel, register_physics
from .newns_anderson import hilbert_multiplier, hilbert_transform


@register_physics("shared_adsorption")
class SharedAdsorptionPhysics(PhysicsModel):
    """Sum effective hybridization/orthogonalization channels and local energy.

    Latent shape is (n_adsorbate_atoms, n_channels, 6): resonance, squared
    coupling (already nonnegative), broadening, band center, width, alpha.
    Physical parameters are the three energy contributions per structure.
    ``channel_parameters`` exposes the intermediate effective-state values.
    """

    n_latent = 6
    constant_names = ("adsorbate_batch", "local_energy")
    parameter_names = ("local_energy", "hybridization", "orthogonalization")
    parameter_labels = ("Learned local energy", "Hybridization", "Orthogonalization")
    property_label = "Energy (eV; training reference convention)"
    channel_parameter_names = (
        "resonance_energy",
        "coupling_squared",
        "broadening",
        "band_center",
        "band_half_width",
        "orthogonalization_coefficient",
    )

    def __init__(self, n_grid=301, energy_limit=20.0, band_smoothing=0.05):
        if n_grid < 51 or n_grid % 2 != 1:
            raise ValueError("n_grid must be odd and at least 51.")
        if not math.isfinite(energy_limit) or energy_limit < 15:
            raise ValueError("energy_limit must be finite and at least 15 eV.")
        if not math.isfinite(band_smoothing) or band_smoothing <= 0:
            raise ValueError("band_smoothing must be positive and finite.")
        self.n_grid = n_grid
        self.energy_limit = float(energy_limit)
        self.band_smoothing = float(band_smoothing)

    @staticmethod
    def channel_parameters(latent):
        return torch.stack(
            (
                8.0 * torch.tanh(latent[..., 0] / 8.0),
                latent[..., 1],
                0.2 + 2.0 * torch.sigmoid(latent[..., 2]),
                6.0 * torch.tanh(latent[..., 3] / 6.0),
                0.5 + 5.0 * torch.sigmoid(latent[..., 4]),
                0.2 * torch.sigmoid(latent[..., 5]),
            ),
            dim=-1,
        )

    def channel_terms(self, params):
        shape = params.shape[:-1]
        eps, coupling, delta, center, width, alpha = params.reshape(-1, 6).unbind(-1)
        energy = torch.linspace(
            -self.energy_limit,
            self.energy_limit,
            self.n_grid,
            dtype=params.dtype,
            device=params.device,
        )
        x = (energy[None, :] - center[:, None]) / width[:, None]
        smooth = self.band_smoothing
        dos = torch.sqrt(F.softplus((1.0 - x.square()) / smooth) * smooth + 1e-12)
        dos = dos / torch.trapz(dos, energy, dim=-1)[:, None]
        occupied = slice(0, self.n_grid // 2 + 1)
        filling = torch.trapz(dos[:, occupied], energy[occupied], dim=-1)
        broadening = math.pi * coupling[:, None] * dos + delta[:, None]
        multiplier = hilbert_multiplier(
            self.n_grid, dtype=params.dtype, device=params.device
        )
        shift = hilbert_transform(broadening, multiplier[None, :])
        phase = (
            torch.atan2(broadening, energy[None, :] - eps[:, None] - shift) - math.pi
        )
        phase_ref = (
            torch.atan2(delta[:, None].expand_as(phase), energy[None, :] - eps[:, None])
            - math.pi
        )
        hybridization = (
            2.0
            / math.pi
            * torch.trapz((phase - phase_ref)[:, occupied], energy[occupied], dim=-1)
        )
        lorentzian = delta[:, None] / (
            math.pi
            * ((energy[None, :] - eps[:, None]).square() + delta[:, None].square())
        )
        population = torch.trapz(lorentzian[:, occupied], energy[occupied], dim=-1)
        orthogonalization = 2.0 * (population + filling) * alpha * coupling
        return {
            "hybridization": hybridization.reshape(shape),
            "orthogonalization": orthogonalization.reshape(shape),
            "filling": filling.reshape(shape),
            "population": population.reshape(shape),
        }

    def parameters(self, latent, constants):
        terms = self.channel_terms(self.channel_parameters(latent))
        local = constants["local_energy"]
        index = constants["adsorbate_batch"]
        contributions = [local]
        for name in ("hybridization", "orthogonalization"):
            per_atom = terms[name].sum(-1)
            contributions.append(torch.zeros_like(local).index_add(0, index, per_atom))
        return torch.stack(contributions, dim=-1)

    def property_from_parameters(self, params):
        return params.sum(-1)
