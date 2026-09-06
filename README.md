# TinNet

TinNet (Theory-infused Neural Network) is a foundation AI package for integrated catalysis.
The `tinnet` module provides pretrained, interpretable models for adsorption energy,
cohesive energy, d-band center, and d-band moments, with SHAP explanations expressed in
terms of physical parameters.

## Usage

```python
from tinnet import tinnet

model = tinnet.AdsorptionEnergy(image=slab, site_inx=[20], adsorbate='OH', name='Pt_MLPt3Co111')
y = model.predict()                                # ensemble adsorption energies (eV)
terms = model.decompose()                          # per-orbital hybridization / orthogonalization terms
fig, ax = model.explain_shap(ref_image=pure_pt, ref_site_inx=[5], ref_name='Pure Pt')
```

See `examples/run_tinnet_example.py` and `tutorials/tinnet.ipynb` for the full workflow
(adsorption energy, band center, band moments, cohesive energy).

## Architecture: one contract for every property

Every TinNet model is a graph neural network followed by a **physics head**:

```
crystal graph --GNN--> latent outputs --PhysicsModel.parameters()--> physical parameters
                                       --PhysicsModel.property_from_parameters()--> property
```

The physics heads live in `tinnet/tinnet/physics/` and share the
`PhysicsModel` contract (`physics/base.py`):

| registry name      | class                  | property                       | latent per row | constants                          |
|--------------------|------------------------|--------------------------------|----------------|------------------------------------|
| `newns_anderson`   | `NewnsAndersonModel`   | adsorption energy              | 3/orbital (+2) | `vad2` (+ `d_cen`, `width`)        |
| `rectangular_band` | `RectangularBandModel` | d-band filling / center / width| M + 3          | hopping couplings, bulk references |
| `moments`          | `MomentModel`          | d-band moments μ2, μ3, μ4      | 3 per pair     | tight-binding hopping masks        |
| `cohesion`         | `CohesionModel`        | cohesive energy per atom       | 6              | promotion energy, Wigner-Seitz volume |

Because `property_from_parameters` is the *only* implementation of each theory equation,
one object serves checkpoint inference, SHAP (`physics.shap_function`), and training
(gradients flow through the theory into the GNN). `physics.describe()` prints the contract.

Supporting modules:

- `gnn.py` – shared CGCNN with `mean`, `site`, or `atom` readout (checkpoint-compatible).
- `descriptors.py` – Voronoi crystal-graph descriptor.
- `training.py` – `TheoryInfusedModel(gnn, physics)`, `fit`, `evaluate`, `save_checkpoint`.

## Extending to new adsorbates

For a shared model that consumes complete adsorbate/surface geometries and
predicts conservative forces, see the new [shared adsorption S2EF guide](docs/shared_adsorption.md).
It includes OC20 LMDB ingestion, energy/force training, and ASE relaxation.
This is an initial trainable architecture; its tutorial checkpoints are not
validated predictors for the full OC20 chemistry space.

The Newns-Anderson head is adsorbate-agnostic. An adsorbate is *data*: which frontier
orbitals couple to the d band, their degeneracy, the orthogonalization coefficient, the sp
offset, the geometry above the site, and which pretrained ensemble to load.

```python
from tinnet.tinnet.physics import AdsorbateSpec, OrbitalSpec, register_adsorbate

register_adsorbate(AdsorbateSpec(
    name='N',
    orbitals=(OrbitalSpec('pz',  r'p_{z}',  degeneracy=1, alpha=0.07),
              OrbitalSpec('pxy', r'p_{xy}', degeneracy=2, alpha=0.05)),
    esp=-4.0,
    geometry=(('N', (0.0, 0.0, 1.55)),),
    d_band='latent',            # GNN predicts d_cen/width (as for OH); 'tabulated' uses BandCenter (as for O)
    gnn=dict(atom_fea_len=64, n_conv=3, h_fea_len=64, n_h=2),
    checkpoint=('adsorption_energy', 'N', 'atop'),
))

model = tinnet.AdsorptionEnergy(image=slab, site_inx=[site], adsorbate='N')
```

OH and O are two such entries (`physics/adsorbates.py`); no theory code changed.
`examples/add_new_adsorbate.py` is a complete template that registers an adsorbate,
trains an ensemble through the theory head with `training.fit`, saves checkpoints where
`AdsorptionEnergy` expects them, and predicts with the facade.

## Extending to new properties

1. Subclass `PhysicsModel`, declare `n_latent`, `constant_names`, `parameter_names`
   (+ labels), and implement `parameters(latent, constants)` and
   `property_from_parameters(params)` as batched, differentiable torch code.
2. Decorate it with `@register_physics('my_property')`.
3. Choose a GNN readout (`mean` per structure, `site` per site, `atom` per atom), build
   `TheoryInfusedModel(gnn, physics)`, and train with `training.fit`. Auxiliary targets on
   any physical parameter (for example a DFT d-band center) are added with
   `parameter_weights={'d_cen': 0.1}`.
4. SHAP comes for free: `shap.Explainer(physics.shap_function, reference_parameters)`.

## Tests

```
pytest                    # fast unit tests of the physics package (no checkpoints needed)
pytest -m integration     # notebook examples against data/pretrained (a few minutes)
```

## Refactor notes (September 2026)

- Added the `physics/` package, `gnn.py`, `descriptors.py`, and `training.py`.
- `adsorption_energy.py` no longer contains OH/O branches: `AdsorptionEnergy` reads an
  `AdsorbateSpec`, and the Newns-Anderson equation exists once (previously a scalar copy for
  inference and a batched copy for SHAP).
- `cohesive_energy.py`, `band_center.py`, and `band_moments.py` delegate their theory
  to `CohesionModel`, `RectangularBandModel`, and `MomentModel`; the duplicated `ConvLayer`
  now comes from `gnn.py`. Cohesive-energy inference no longer spawns a `multiprocessing.Pool`
  (which re-imported the calling script on macOS).
- Validation: all notebook quantities were recomputed with the original code and the
  refactored code. Band center, band moments, and cohesive energies are bit-identical;
  adsorption energies agree to 4e-6 eV; SHAP values agree to 1e-4 eV (the band-center SHAP
  uses a stochastic permutation explainer and varies run to run by the same amount).
- Removed: the in-module `Regression`/`Chemisorption`/`TightBinding` classes (replaced by
  `AdsorptionEnsemble`/`CohesionEnsemble` plus the physics heads). Public notebook APIs are unchanged.
