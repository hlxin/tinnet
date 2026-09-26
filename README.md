# Tinnet

Tinnet is a foundation AI package for integrated catalysis. The first released module is `tinnet` for property prediction with pretrained models for adsorption energy, cohesive energy, band center, and band moments. SHAP explainability is provided.

## Installation

```bash
git clone https://github.com/hlxin/tinnet.git
cd tinnet
pip install -e .
```

The pretrained models are read from the repository's `data/` directory, so an editable install (`-e`) is recommended. To keep the data elsewhere, set the `TINNET_DATA_DIR` environment variable to that directory.

## Usage

Each model takes an ASE `Atoms` object when it is created; `predict()` and `explain_shap()` then work on that structure.

```python
from ase.build import fcc111
from tinnet import AdsorptionEnergy, BandCenter, BandMoments, CohesiveEnergy

# Cu single-atom alloy in an Ag(111) slab
slab = fcc111('Ag', size=(2, 2, 4), a=4.15)
slab.center(vacuum=7.5, axis=2)
slab[-1].symbol = 'Cu'

model = AdsorptionEnergy(image=slab, site_inx=[15], adsorbate='O', name='Cu1Ag SAA')
model.predict()

ref = fcc111('Cu', size=(2, 2, 4), a=3.63)
ref.center(vacuum=7.5, axis=2)
model.explain_shap(ref_image=ref, ref_site_inx=[15], ref_name='Pure Cu', save_fig='png')

BandCenter(image=slab, atom_inx=15, name='Cu1Ag SAA').predict()
BandMoments(image=slab, atom_inx=15, name='Cu1Ag SAA').predict()
```

`AdsorptionEnergy` supports `adsorbate='O'` and `adsorbate='OH'` on atop sites. See [`tutorials/tinnet.ipynb`](tutorials/tinnet.ipynb) and [`examples/run_tinnet_example.py`](examples/run_tinnet_example.py) for complete examples.

## Tests

```bash
pip install -e ".[dev]"
pytest tests
```

## License

MIT, see [LICENSE](LICENSE).
