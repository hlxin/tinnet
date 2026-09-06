"""Physical symmetries, force differentiation, data conventions, and training."""

import pickle

import numpy as np
import pytest
import torch
from ase import Atoms

from tinnet.tinnet.ase_calculator import SharedAdsorptionCalculator, relax_adsorption
from tinnet.tinnet.oc20 import OC20LmdbDataset, oc20_sample
from tinnet.tinnet.physics.shared_adsorption import SharedAdsorptionPhysics
from tinnet.tinnet.s2ef import (
    evaluate_s2ef,
    fit_s2ef,
    load_shared_checkpoint,
    s2ef_loss,
    save_shared_checkpoint,
)
from tinnet.tinnet.shared_adsorption import SharedAdsorptionModel
from tinnet.tinnet.structures import configuration_graph, collate_configurations


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def atoms():
    # Oxygen occurs in both the slab and adsorbate; tags, not elements, separate them.
    return Atoms(
        "PtPtOCOH",
        positions=[
            [-1.3, 0, 0],
            [1.3, 0, 0],
            [0, 1.8, 0],
            [0.3, 0.2, 1.7],
            [1.0, 0, 2.5],
            [0.6, 1, 2.8],
        ],
        cell=[7, 8, 10],
        pbc=[True, True, False],
        tags=[0, 1, 1, 2, 2, 2],
    )


@pytest.fixture
def model():
    torch.manual_seed(12)
    return (
        SharedAdsorptionModel(
            hidden=12,
            n_layers=2,
            n_rbf=8,
            n_grid=101,
            cutoff=4.0,
            energy_reference="synthetic",
        )
        .double()
        .eval()
    )


def graph(atoms):
    return configuration_graph(atoms, cutoff=4.0, dtype=torch.float64)


def test_masks_preserve_geometry_and_slab_elements(atoms):
    before = atoms.copy()
    g = graph(atoms)
    assert g["numbers"].tolist() == atoms.numbers.tolist()
    assert g["adsorbate_mask"].tolist() == [False, False, False, True, True, True]
    assert g["fixed_mask"].tolist() == [True, False, False, False, False, False]
    np.testing.assert_array_equal(g["positions"].numpy(), before.positions)
    np.testing.assert_array_equal(atoms.positions, before.positions)
    atoms.set_tags([0] * len(atoms))
    with pytest.raises(ValueError, match="Supply adsorbate_mask"):
        graph(atoms)
    explicit = configuration_graph(atoms, adsorbate_mask=[0, 0, 0, 1, 1, 1])
    assert not explicit["fixed_mask"].any()
    with pytest.raises(ValueError, match="both adsorbate and slab"):
        configuration_graph(atoms, adsorbate_mask=[1] * len(atoms))


def test_energy_decomposes_and_forces_match_finite_differences(model, atoms):
    output = model(graph(atoms), compute_forces=True)
    torch.testing.assert_close(output["energy"], output["parameters"].sum(-1))
    step = 1e-5
    numerical = np.empty((len(atoms), 3))
    # Rebuild the graph for every displaced geometry, as ASE relaxation does.
    for i in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[i, axis] += step
            minus.positions[i, axis] -= step
            with torch.no_grad():
                numerical[i, axis] = -(
                    model(graph(plus))["energy"].item()
                    - model(graph(minus))["energy"].item()
                ) / (2 * step)
    np.testing.assert_allclose(
        output["forces"].detach(), numerical, rtol=2e-5, atol=2e-6
    )


def test_rotation_translation_permutation_and_periodicity(model, atoms):
    original = model(graph(atoms), compute_forces=True)
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    )
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T + [2.3, -1.2, 4.1]
    rotated.set_cell(atoms.cell.array @ rotation.T)
    output = model(graph(rotated), compute_forces=True)
    torch.testing.assert_close(
        output["energy"], original["energy"], rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        output["forces"].detach(),
        original["forces"].detach().numpy() @ rotation.T,
        rtol=1e-8,
        atol=1e-8,
    )
    order = [5, 2, 0, 4, 1, 3]
    permuted = model(graph(atoms[order]), compute_forces=True)
    torch.testing.assert_close(
        permuted["energy"], original["energy"], rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        permuted["forces"], original["forces"][order], rtol=1e-8, atol=1e-8
    )
    shifted = atoms.copy()
    shifted.positions[3] += atoms.cell.array[0]
    periodic = model(graph(shifted), compute_forces=True)
    torch.testing.assert_close(
        periodic["energy"], original["energy"], rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        periodic["forces"], original["forces"], rtol=1e-8, atol=1e-8
    )
    torch.testing.assert_close(
        original["forces"].sum(0),
        torch.zeros(3, dtype=torch.float64),
        atol=1e-9,
        rtol=0,
    )


def test_variable_adsorbates_batched_match_individual_predictions(model, atoms):
    other = atoms[:4]
    batch = collate_configurations([{"graph": graph(atoms)}, {"graph": graph(other)}])
    output = model(batch["graph"], compute_forces=True)
    individual = [model(graph(a), compute_forces=True) for a in (atoms, other)]
    torch.testing.assert_close(
        output["energy"], torch.cat([o["energy"] for o in individual])
    )
    torch.testing.assert_close(
        output["forces"], torch.cat([o["forces"] for o in individual])
    )
    assert output["channel_parameters"].shape == (4, 2, 6)


def test_zero_contact_and_cutoff_continuity(model):
    far = Atoms("PtH", positions=[[0, 0, 0], [5, 0, 0]], tags=[0, 2])
    output = model(graph(far), compute_forces=True)
    torch.testing.assert_close(
        output["parameters"][:, 1:],
        torch.zeros((1, 2), dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    torch.testing.assert_close(
        output["forces"], torch.zeros((2, 3), dtype=torch.float64), atol=1e-10, rtol=0
    )
    energies, forces = [], []
    for distance in (4.0 - 1e-5, 4.0 + 1e-5):
        far.positions[1, 0] = distance
        out = model(graph(far), compute_forces=True)
        energies.append(out["energy"])
        forces.append(out["forces"])
    torch.testing.assert_close(energies[0], energies[1], atol=1e-8, rtol=0)
    torch.testing.assert_close(forces[0], forces[1], atol=1e-7, rtol=0)


def test_channel_physics_zero_coupling_has_finite_second_derivatives():
    physics = SharedAdsorptionPhysics(n_grid=101)
    latent = torch.tensor(
        [[[-2.0, 0.0, 0.0, -1.0, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True
    )
    energy, terms = physics.forward(
        latent,
        {
            "local_energy": torch.tensor([1.0], dtype=torch.float64),
            "adsorbate_batch": torch.tensor([0]),
        },
    )
    torch.testing.assert_close(energy, torch.ones_like(energy), atol=1e-10, rtol=0)
    first = torch.autograd.grad(energy.sum(), latent, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), latent)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def record(atoms):
    return dict(
        atomic_numbers=atoms.numbers,
        pos=atoms.positions,
        cell=atoms.cell.array[None],
        pbc=atoms.pbc,
        tags=atoms.get_tags(),
        y=-7.0,
        force=np.zeros((len(atoms), 3)),
        sid=42,
        fid=7,
    )


def test_oc20_reference_conversion_and_s2ef_label_validation(atoms):
    data = record(atoms)
    sample = oc20_sample(data, energy_reference="total", reference_energy=-10.0)
    assert sample["energy"].item() == 3.0
    assert sample["energy_reference"] == "oc20-referenced"
    assert sample["sid"] == 42 and sample["fid"] == 7
    with pytest.raises(ValueError, match="Reference subtraction"):
        oc20_sample(data, energy_reference="oc20-referenced", reference_energy=-10.0)
    bad = {**data, "y": None, "y_relaxed": -3.0}
    with pytest.raises(ValueError, match="IS2RE"):
        oc20_sample(bad, energy_reference="oc20-referenced")
    with pytest.raises(ValueError, match="mix energy reference"):
        collate_configurations([sample, oc20_sample(data, energy_reference="total")])
    with pytest.raises(ValueError, match="Missing forces"):
        collate_configurations(
            [sample, {k: v for k, v in sample.items() if k != "forces"}]
        )


def test_lmdb_shards_and_legacy_pyg_records(tmp_path, atoms):
    lmdb = pytest.importorskip("lmdb")
    pyg = pytest.importorskip("torch_geometric.data")
    legacy = pyg.Data()
    legacy.__dict__.clear()
    legacy.__dict__.update(record(atoms))
    for i, value in enumerate((record(atoms), legacy)):
        with lmdb.open(
            str(tmp_path / f"{i}.lmdb"), subdir=False, map_size=1 << 20
        ) as env:
            with env.begin(write=True) as txn:
                txn.put(b"0", pickle.dumps(value))
                txn.put(b"length", pickle.dumps(1))
    dataset = OC20LmdbDataset(tmp_path, energy_reference="oc20-referenced")
    assert len(dataset) == 2
    assert dataset[0]["energy"] == dataset[-1]["energy"]
    assert dataset[1]["sid"] == 42
    with pytest.raises(IndexError):
        dataset[2]
    assert pickle.loads(pickle.dumps(dataset))._envs == {}
    dataset.close()


def spring_batch():
    samples = []
    for symbol in ("H", "O"):
        for r in np.linspace(1.5, 2.6, 5):
            atoms = Atoms(["Pt", symbol], positions=[[0, 0, 0], [0, 0, r]], tags=[0, 2])
            force = np.array([[0, 0, 1.2 * (r - 2.0)], [0, 0, -1.2 * (r - 2.0)]])
            samples.append(
                {
                    "graph": configuration_graph(atoms, cutoff=4.0),
                    "energy": -1.5 + 0.6 * (r - 2.0) ** 2,
                    "forces": force,
                    "energy_reference": "synthetic",
                }
            )
    return collate_configurations(samples)


def test_energy_force_training_reduces_independent_analytic_target_errors():
    torch.manual_seed(4)
    model = SharedAdsorptionModel(
        hidden=16,
        n_layers=2,
        n_rbf=12,
        n_grid=101,
        cutoff=4.0,
        energy_reference="synthetic",
    )
    batch = spring_batch()
    before = evaluate_s2ef(model, [batch])
    fit_s2ef(model, [batch], epochs=60, lr=3e-3, log_every=0)
    after = evaluate_s2ef(model, [batch])
    assert after["energy_mae"] < before["energy_mae"] * 0.5
    assert after["force_mae"] < before["force_mae"] * 0.5


def test_fixed_atom_labels_do_not_affect_force_loss(model, atoms):
    sample = oc20_sample(
        record(atoms), energy_reference="synthetic", cutoff=4.0, dtype=torch.float64
    )
    batch = collate_configurations([sample])
    original, _ = s2ef_loss(model, batch, energy_weight=0)
    batch["forces"][batch["graph"]["fixed_mask"]] = 1e6
    changed, _ = s2ef_loss(model, batch, energy_weight=0)
    torch.testing.assert_close(original, changed)
    changed.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_checkpoint_ase_and_fixed_atom_relaxation(tmp_path, model, atoms):
    path = save_shared_checkpoint(model, tmp_path / "model.pt")
    loaded = load_shared_checkpoint(path)
    a = model(graph(atoms), compute_forces=True)
    b = loaded(graph(atoms), compute_forces=True)
    torch.testing.assert_close(a["energy"], b["energy"], rtol=0, atol=0)
    torch.testing.assert_close(a["forces"], b["forces"], rtol=0, atol=0)
    atoms.calc = SharedAdsorptionCalculator(loaded)
    assert atoms.get_potential_energy() == b["energy"].item()
    np.testing.assert_allclose(atoms.get_forces(), b["forces"].detach().numpy())
    before = atoms.positions.copy()
    result = relax_adsorption(atoms, loaded, steps=2, fmax=1e-9)
    np.testing.assert_array_equal(atoms.positions, before)
    np.testing.assert_array_equal(result["atoms"].positions[0], before[0])
    assert np.isfinite(result["atoms"].get_potential_energy())
    assert result["steps"] == 2
