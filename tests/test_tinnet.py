import os

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

pytest.importorskip('torch')
pytest.importorskip('ase')
pytest.importorskip('pymatgen')
pytest.importorskip('shap')

from ase.build import fcc111

from tinnet import AdsorptionEnergy, BandCenter, BandMoments, CohesiveEnergy
from tinnet._paths import data_path


def slab(host, a, size=(2, 2, 4), guests=()):
    image = fcc111(host, size=size, a=a)
    image.center(vacuum=7.5, axis=2)
    for i in guests:
        image[i].symbol = 'Cu'
    return image


@pytest.fixture(autouse=True)
def run_elsewhere(tmp_path, monkeypatch):
    # Models must load no matter what the working directory is.
    monkeypatch.chdir(tmp_path)


def test_pretrained_data_present():
    assert os.path.isfile(data_path('MaterialDict.pkl'))
    for model in ['adsorption_energy/O/atop', 'adsorption_energy/OH/atop',
                  'band_center', 'band_moments', 'cohesive_energy']:
        for i in range(10):
            assert os.path.isfile(data_path('pretrained', model, 'model_%d.pth.tar' % i))


def test_adsorption_energy():
    model = AdsorptionEnergy(image=slab('Ag', 4.15, guests=[-1]), site_inx=[15], adsorbate='O')
    assert model.predict() is not None


def test_band_center():
    model = BandCenter(image=slab('Ag', 4.15, guests=[-1]), atom_inx=15)
    assert model.predict() is not None


def test_band_moments():
    model = BandMoments(image=slab('Ag', 4.15, guests=[-1]), atom_inx=15)
    assert model.predict() is not None


def test_cohesive_energy():
    model = CohesiveEnergy(image=slab('Ag', 4.15, size=(4, 4, 4), guests=[-1, -2]))
    assert model.predict() is not None


def test_adsorption_energy_shap():
    model = AdsorptionEnergy(image=slab('Ag', 4.15, guests=[-1]), site_inx=[15], adsorbate='O')
    model.explain_shap(ref_image=slab('Cu', 3.63), ref_site_inx=[15], ref_name='Pure Cu', save_fig='png')


def test_band_center_shap():
    model = BandCenter(image=slab('Ag', 4.15, guests=[-1]), atom_inx=15)
    model.explain_shap(ref_image=slab('Cu', 3.63), ref_atom_inx=15, ref_name='Pure Cu', save_fig='png')


def test_cohesive_energy_shap():
    model = CohesiveEnergy(image=slab('Ag', 4.15, size=(4, 4, 4), guests=[-1, -2]))
    model.explain_shap(ref_image=slab('Ag', 4.15, size=(4, 4, 4), guests=[-1, -3]),
                       ref_name='Cu_di_saa_Ag', save_fig='png')
