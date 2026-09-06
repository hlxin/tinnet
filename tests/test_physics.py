"""Unit tests for the TinNet physics (theory-module) package.

These tests need torch/numpy only; they do not touch the pretrained
checkpoints or pymatgen, so they run in a few seconds.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinnet.tinnet.physics import (  # noqa: E402
    AdsorbateSpec,
    CohesionModel,
    MomentModel,
    NewnsAndersonModel,
    OrbitalSpec,
    RectangularBandModel,
    adsorbate_spec,
    available_adsorbates,
    available_physics,
    get_physics,
    hopping_couplings,
    register_adsorbate,
    semi_ellipse_dos,
)
from tinnet.tinnet.physics.adsorbates import ADSORBATE_REGISTRY  # noqa: E402
from tinnet.tinnet import gnn, training  # noqa: E402


# ----------------------------------------------------------------------
# Registry / specs
# ----------------------------------------------------------------------
def test_registry_contains_builtin_models():
    assert {"newns_anderson", "cohesion", "rectangular_band", "moments"} <= set(available_physics())
    assert {"OH", "O"} <= set(available_adsorbates())
    assert isinstance(get_physics("newns_anderson", spec="OH"), NewnsAndersonModel)


def test_builtin_adsorbate_layouts_match_published_checkpoints():
    oh = adsorbate_spec("OH")
    assert oh.n_latent == 11 and oh.constant_names == ("vad2",)
    assert oh.parameter_names == (
        "vad2", "d_cen", "width",
        "adse_3sigma", "beta_3sigma", "delta_3sigma",
        "adse_1pi", "beta_1pi", "delta_1pi",
        "adse_4sigma*", "beta_4sigma*", "delta_4sigma*",
    )
    o = adsorbate_spec(phys_model="O_atop")
    assert o.n_latent == 6 and o.constant_names == ("vad2", "d_cen", "width")
    assert [orb.degeneracy for orb in o.orbitals] == [1, 2]


def test_register_new_adsorbate_is_data_only():
    spec = AdsorbateSpec(
        name="N_test",
        orbitals=(OrbitalSpec("pz", r"p_{z}", 1, 0.07), OrbitalSpec("pxy", r"p_{xy}", 2, 0.05)),
        esp=-4.0,
        geometry=(("N", (0.0, 0.0, 1.5)),),
        d_band="tabulated",
    )
    try:
        register_adsorbate(spec)
        with pytest.raises(ValueError):
            register_adsorbate(spec)  # duplicate without overwrite
        physics = NewnsAndersonModel("N_test")
        assert physics.n_latent == 6
        assert physics.constant_names == ("vad2", "d_cen", "width")
        assert spec.elements == frozenset({"N"})
        latent = torch.randn(3, physics.n_latent)
        energy, params = physics(latent, {"vad2": torch.full((3,), 3.0),
                                          "d_cen": torch.full((3,), -2.0),
                                          "width": torch.full((3,), 3.0)})
        assert energy.shape == (3,) and params.shape == (3, physics.n_parameters)
        assert torch.isfinite(energy).all()
    finally:
        ADSORBATE_REGISTRY.pop("N_test", None)


def test_spec_validation():
    with pytest.raises(ValueError):
        AdsorbateSpec(name="bad", orbitals=(), esp=0.0, geometry=(("O", (0, 0, 1)),))
    with pytest.raises(ValueError):
        AdsorbateSpec(name="bad", orbitals=(OrbitalSpec("a", "a"),), esp=0.0,
                      geometry=(("O", (0, 0, 1)),), d_band="unknown")


# ----------------------------------------------------------------------
# Newns-Anderson theory
# ----------------------------------------------------------------------
def test_semi_ellipse_dos_is_normalized_and_centered():
    ergy = torch.linspace(-15, 15, 3001)
    d_cen = torch.tensor([-2.0, 1.0])
    width = torch.tensor([3.0, 2.0])
    dos = semi_ellipse_dos(ergy, d_cen, width)
    area = torch.trapz(dos, ergy, dim=1)
    center = torch.trapz(dos * ergy, ergy, dim=1)
    assert torch.allclose(area, torch.ones(2), atol=1e-5)
    assert torch.allclose(center, d_cen, atol=1e-2)


def test_newns_anderson_zero_coupling_reduces_to_esp():
    """With beta = 0 the hybridization and orthogonalization terms vanish."""
    physics = NewnsAndersonModel("OH")
    params = torch.zeros(1, physics.n_parameters)
    params[0, 0] = 2.0        # vad2
    params[0, 1] = -2.0       # d_cen
    params[0, 2] = 2.0        # width
    for k in range(physics.spec.n_orbitals):
        params[0, 3 + 3 * k] = -5.0  # adse
        params[0, 4 + 3 * k] = 0.0   # beta = 0
        params[0, 5 + 3 * k] = 0.5   # delta
    energy = physics.property_from_parameters(params)
    assert math.isclose(float(energy), physics.spec.esp, abs_tol=1e-5)


def test_newns_anderson_batched_matches_row_by_row():
    physics = NewnsAndersonModel("OH")
    torch.manual_seed(0)
    latent = torch.randn(5, physics.n_latent)
    vad2 = torch.rand(5) * 5 + 1
    energy, params = physics(latent, {"vad2": vad2})
    for i in range(5):
        e_i, _ = physics(latent[i: i + 1], {"vad2": vad2[i: i + 1]})
        assert torch.allclose(e_i, energy[i: i + 1], atol=1e-5)
    # SHAP wrapper is the same function on NumPy input
    np.testing.assert_allclose(physics.shap_function(params.numpy()), energy.numpy(), atol=1e-5)


def test_newns_anderson_is_differentiable_and_decomposes():
    physics = NewnsAndersonModel("O")
    latent = torch.randn(2, physics.n_latent, requires_grad=True)
    constants = {"vad2": torch.tensor([2.0, 3.0]), "d_cen": torch.tensor([-2.0, -1.5]), "width": torch.tensor([2.5, 3.0])}
    energy, params = physics(latent, constants)
    energy.sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()

    terms = physics.decompose(params.detach())
    total = torch.full_like(energy, physics.spec.esp)
    for orbital in physics.spec.orbitals:
        total = total + terms[f"contribution_{orbital.name}"]
    assert torch.allclose(total, energy.detach(), atol=1e-5)
    ergy, dos = physics.projected_dos(params.detach().numpy(), orbital="pz")
    assert ergy.shape == (3001,) and dos.shape == (2, 3001)


# ----------------------------------------------------------------------
# Cohesion
# ----------------------------------------------------------------------
def test_cohesion_parameters_sum_to_energy():
    physics = CohesionModel()
    latent = torch.randn(4, physics.n_latent)
    constants = {"promotion_energy": torch.rand(4), "wigner_seitz_volume": torch.rand(4) * 10 + 10}
    energy, params = physics(latent, constants)
    assert params.shape == (4, 4)
    assert torch.allclose(energy, params.sum(dim=1))
    assert torch.allclose(params[:, 0], constants["promotion_energy"])


# ----------------------------------------------------------------------
# Rectangular band
# ----------------------------------------------------------------------
def test_rectangular_band_fast_path_equals_parameter_path():
    M = 5
    physics = RectangularBandModel(max_neighbors=M)
    torch.manual_seed(1)
    latent = torch.randn(2, physics.n_latent, dtype=torch.float64)
    site_radius = torch.tensor([0.7, 0.8], dtype=torch.float64)
    neighbor_radii = torch.rand(2, M, dtype=torch.float64) + 0.5
    distances = torch.rand(2, M, dtype=torch.float64) + 2.5
    padding = torch.ones(2, M, dtype=torch.float64)
    padding[1, -2:] = 0
    v2ds, v2dd = hopping_couplings(site_radius[:, None], neighbor_radii, distances)
    constants = dict(site_radius=site_radius, neighbor_radii=neighbor_radii, distances=distances,
                     bulk_center=torch.tensor([-2.5, -1.5], dtype=torch.float64),
                     bulk_width=torch.tensor([4.0, 6.0], dtype=torch.float64),
                     mulliken=torch.tensor([0.1, -0.2], dtype=torch.float64),
                     padding_filter=padding, v2ds=v2ds * padding, v2dd=v2dd * padding)
    fast = physics.band_properties(latent, constants)
    d_cen, params = physics(latent, constants)
    assert params.shape == (2, 4 * M + 6)
    assert torch.allclose(d_cen, fast["d_cen"], rtol=1e-10)
    effects = physics.aggregate_effects(np.ones((1, params.shape[1])))
    # beta (1) + M distances + M zeta + M radii + alpha & mulliken (2)
    assert effects.shape == (5,) and effects.sum() == 3 * M + 3


# ----------------------------------------------------------------------
# Moments
# ----------------------------------------------------------------------
def test_moments_of_symmetric_hopping_matrix():
    physics = MomentModel()
    P = 3
    h = torch.zeros(1, P, 6, P, 6, dtype=torch.float64)
    # Coupling of site d orbitals to one neighbor with hopping t, d-only channels.
    t = 0.5
    for o in range(1, 6):
        h[0, 0, o, 1, o] = t
        h[0, 1, o, 0, o] = t
    moments = physics.moments(h)
    assert moments.shape == (1, 3)
    assert math.isclose(float(moments[0, 0]), 5 * t ** 2, rel_tol=1e-12)   # m2 = sum_d t^2
    assert math.isclose(float(moments[0, 1]), 0.0, abs_tol=1e-12)          # bipartite -> m3 = 0
    assert math.isclose(float(moments[0, 2]), 5 * t ** 4, rel_tol=1e-12)


# ----------------------------------------------------------------------
# Training scaffold: learn through the theory equation
# ----------------------------------------------------------------------
def test_theory_infused_training_reduces_loss():
    """A small MLP 'encoder' trained through the Newns-Anderson head on synthetic data."""
    torch.manual_seed(0)
    physics = NewnsAndersonModel("O")

    # Synthetic ground truth: a hidden linear map from features to latent outputs.
    n, n_feat = 32, 4
    features = torch.randn(n, n_feat)
    true_latent = features @ torch.randn(n_feat, physics.n_latent) * 0.5 + torch.tensor([-4, 0, 0, -3, 0, 0.0])
    constants = {"vad2": torch.full((n,), 2.5), "d_cen": torch.full((n,), -2.0), "width": torch.full((n,), 3.0)}
    with torch.no_grad():
        target, _ = physics(true_latent, constants)

    encoder = torch.nn.Sequential(torch.nn.Linear(n_feat, 16), torch.nn.Softplus(), torch.nn.Linear(16, physics.n_latent))
    model = training.TheoryInfusedModel(encoder, physics)
    batches = [{"inputs": (features,), "constants": constants, "target": target,
                "parameter_targets": {"d_cen": constants["d_cen"]}}]
    before = training.evaluate(model, batches)
    history = training.fit(model, batches, epochs=25, lr=1e-2, parameter_weights={"d_cen": 0.1})
    after = training.evaluate(model, batches)
    assert len(history["train"]) == 25
    assert after < before


def test_cgcnn_readouts_share_checkpoint_layout():
    """The shared CGCNN exposes the same parameter names for every readout."""
    torch.manual_seed(0)
    n_atoms, M, F, K = 6, 3, 5, 4
    atom_fea = torch.randn(n_atoms, F)
    nbr_fea = torch.randn(n_atoms, M, K)
    nbr_idx = torch.randint(0, n_atoms, (n_atoms, M))
    crystal_idx = [torch.arange(n_atoms)]
    nets = {r: gnn.CrystalGraphConvNet(F, K, atom_fea_len=8, n_conv=2, h_fea_len=8, n_h=2, n_out=3, readout=r)
            for r in ("mean", "site", "atom")}
    keys = {r: set(net.state_dict()) for r, net in nets.items()}
    assert keys["mean"] == keys["site"] == keys["atom"]
    for net in nets.values():
        net.eval()
    assert nets["mean"](atom_fea, nbr_fea, nbr_idx, crystal_idx).shape == (1, 3)
    assert nets["site"](atom_fea, nbr_fea, nbr_idx, crystal_idx, site_idx=[2]).shape == (1, 3)
    assert nets["atom"](atom_fea, nbr_fea, nbr_idx).shape == (n_atoms, 3)
    # masked convolution path
    mask = torch.ones(n_atoms, M)
    mask[:, -1] = 0
    assert nets["atom"](atom_fea, nbr_fea, nbr_idx, padding_filter=mask).shape == (n_atoms, 3)
