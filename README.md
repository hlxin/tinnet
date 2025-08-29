# Tinnet

Tinnet is a foundation AI package for integrated catalysis. The first released module is `tinnet` for property prediction with pretrained models for adsorption energy, cohesive energy, band center, and band moments. SHAP explainability is provided.

## Usage

```python
from tinnet import tinnet

model = tinnet.AdsorptionEnergy()
y = model.predict(atoms)
shap_vals = model.explain_shap(atoms)