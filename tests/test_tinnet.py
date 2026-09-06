"""Integration tests against the pretrained TinNet checkpoints.

These reproduce the notebook examples and compare with values recorded from
the original (pre-refactor) implementation.  They need ``data/pretrained``
and pymatgen, and take a few minutes; run with ``pytest -m integration``.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.integration

pytest.importorskip("pymatgen")
from ase import io  # noqa: E402
from ase.build.surface import fcc111  # noqa: E402

from tinnet import tinnet as T  # noqa: E402


@pytest.fixture(scope="module")
def structures():
    if not (ROOT / "data" / "pretrained").exists():
        pytest.skip("pretrained checkpoints not available")
    with open(ROOT / "data" / "MaterialDict.pkl", "rb") as f:
        data = pickle.load(f, encoding="bytes")
    data = {k.decode("utf8"): v for k, v in data.items()}

    def slab(element, size=(2, 2, 4)):
        s = fcc111(element, size=size, a=data[element][b"qLattConst-PBE"])
        s.center(vacuum=7.5, axis=2)
        return s

    pure_cu = slab("Cu")
    cu_ag_saa = slab("Ag")
    cu_ag_saa[-1].symbol = "Cu"
    cu_dimer_ag = slab("Ag", (4, 4, 4))
    cu_dimer_ag[-1].symbol = "Cu"
    cu_dimer_ag[-2].symbol = "Cu"
    images = io.read(ROOT / "data" / "OH_images.traj", index=slice(None))
    return dict(pure_cu=pure_cu, cu_ag_saa=cu_ag_saa, cu_dimer_ag=cu_dimer_ag,
                pure_pt=images[0], pt_mlpt3co=images[1])


# Ensemble means/stds recorded from the original implementation (see notebook).
REFERENCE = {
    "OH_atop_Pt_MLPt3Co111": (-1.848, 0.04),
    "O_atop_Cu1Ag": (-3.24, 0.04),
    "band_center_Cu1Ag": (-1.97, 0.02),
    "cohesive_Cu_dimer_Ag": (-5.65, 0.02),
    "moments_Cu1Ag": ((0.85, 0.56, 11.98), (0.03, 0.03, 0.59)),
}


def test_oh_atop_adsorption(structures):
    model = T.AdsorptionEnergy(image=structures["pt_mlpt3co"], site_inx=[20], adsorbate="OH")
    ead = model.predict()
    parm = model.predict(return_all_parm=True)
    assert ead.shape == (10,) and parm.shape == (10, 1 + model.physics.n_parameters)
    np.testing.assert_allclose(ead.mean(), REFERENCE["OH_atop_Pt_MLPt3Co111"][0], atol=5e-3)
    # column 0 of the parameter matrix is the energy and the theory reproduces it
    np.testing.assert_allclose(model.tinnet_ead(parm[:, 1:]), parm[:, 0], atol=1e-4)


def test_o_atop_adsorption_uses_tabulated_d_band(structures):
    model = T.AdsorptionEnergy(image=structures["cu_ag_saa"], site_inx=[15], adsorbate="O")
    ead = model.predict()
    np.testing.assert_allclose(ead.mean(), REFERENCE["O_atop_Cu1Ag"][0], atol=1e-2)
    assert model.spec.d_band == "tabulated"


def test_band_center(structures):
    model = T.BandCenter(image=structures["cu_ag_saa"], atom_inx=15)
    center = model.predict()
    np.testing.assert_allclose(center.mean(), REFERENCE["band_center_Cu1Ag"][0], atol=1e-2)


def test_cohesive_energy(structures):
    model = T.CohesiveEnergy(image=structures["cu_dimer_ag"])
    ech = model.predict()
    parm = model.predict(return_all_parm=True)
    np.testing.assert_allclose(ech.mean(), REFERENCE["cohesive_Cu_dimer_Ag"][0], atol=1e-2)
    np.testing.assert_allclose(parm[:, 1:].sum(axis=1), parm[:, 0], atol=1e-4)


def test_band_moments(structures):
    model = T.BandMoments(image=structures["cu_ag_saa"], atom_inx=15)
    moments = model.predict()
    assert moments.shape == (10, 3)
    tolerance = np.array([0.02, 0.02, 0.2])
    assert np.all(np.abs(moments.mean(axis=0) - REFERENCE["moments_Cu1Ag"][0]) < tolerance)
