# Tinnet

Tinnet is a foundation AI package for integrated catalysis. The first released module is `tinnet` for property prediction with pretrained models for adsorption energy, cohesive energy, band center, and band moments. SHAP explainability is provided.

## Usage

```python
from tinnet import tinnet

model = tinnet.AdsorptionEnergy()
y = model.predict(atoms)
shap_vals = model.explain_shap(atoms)

Updates
# May 05, 2026

# TinNet refactor notes

# TinNet refactor v2

This version keeps the notebook-facing public APIs and pretrained checkpoint paths intact while reducing duplicated boilerplate across the four model scripts.

## Main engineering changes

- Added shared helpers in `tinnet_utils.py` for:
  - one-pass inference `DataLoader` creation
  - sparse/dense tensor stacking by ids
  - device-safe tensor/list movement
  - signed SHAP labels/colors
  - generic horizontal waterfall plotting
  - figure saving
- `adsorption_energy.py`
  - merged duplicate OH/O checkpoint evaluation paths in `Regression.eval_model()`
  - removed the empty `Features.dict_atom_prop_default()` wrapper
  - reused shared figure-saving and sign-format helpers
- `band_center.py`
  - removed the empty `Features.material_dict()` wrapper
  - `BandCenter.__init__()` now directly calls `material_properties()`
  - merged duplicated `Prediction.predict_d_cen()` and `Prediction.predict_properties()` internals through `_predict_moment_outputs()`
  - reused shared figure-saving and sign-format helpers
- `band_moments.py`
  - moved ensemble defaults to class-level constants
  - removed unused intermediate/debug accumulators from `predict()`
  - extracted ensemble execution and moment-summary printing into helper methods
  - kept the tight-binding hopping construction numerically unchanged
- `cohesive_energy.py`
  - retained the v1 cleanup and removed unused legacy imports

## Validation performed here

- Python syntax check: `python -m py_compile *.py`

Full numerical validation still needs to be run in the actual TinNet environment with `data/pretrained/` and the notebook examples.

# TinNet refactor v1

This folder contains a refactored version of the current TinNet model scripts.
The refactor preserves the trained model architectures, checkpoint filenames,
and physics equations, while cleaning the engineering layer around them.

## Files

- `tinnet_utils.py`: shared utilities for device handling, checkpoint paths,
  non-mutating ASE structure copying, element-constant lookup, and Matplotlib
  publication style.
- `adsorption_energy.py`: OH/O atop adsorption-energy TinNet model.
- `band_center.py`: d-band filling, center, and rectangular-width TinNet model.
- `band_moments.py`: second-, third-, and fourth-order d-band moment TinNet model.
- `cohesive_energy.py`: cohesive-energy TinNet model.
- `tinnet.ipynb`: original notebook kept as a reference driver.

## Design principles

1. Keep the public API used by the notebook unchanged.
2. Do not alter neural-network architectures or physics equations.
3. Avoid mutating the caller's ASE `Atoms` object.
4. Centralize recurring infrastructure code in `tinnet_utils.py`.
5. Use explicit checkpoint-path resolution instead of raw `./data/...` strings.
6. Remove deprecated PyTorch `Variable` usage.
7. Use safer tensor constructors and detach returned analysis parameters when
   they should not keep an autograd graph.

## Validation performed here

- Python syntax compilation for all `.py` files.

## Validation still recommended in your repository

Run the notebook from your TinNet repository root, where the `data/` directory
and pretrained checkpoints are available. Compare the ensemble means and standard
deviations against your current working version for the example systems.
