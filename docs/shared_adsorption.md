# Shared adsorption energy and forces: first implementation

This is a trainable research model for complete adsorbate/surface geometries.
It supports H/C/N/O adsorbates of variable size, multiple adsorbate binding
atoms, and multiple neighboring surface atoms. There are no per-adsorbate
checkpoints, prescribed placements, or atop-only indices in this path.
Existing `AdsorptionEnergy` and its pretrained OH/O models are unchanged.

The new model has no production pretrained weights. Training on the tiny
tutorial fixture checks execution, not coverage of all 82 OC20 adsorbates,
transfer accuracy, or reliability of a relaxed structure.

## Architecture

1. `configuration_graph` builds a periodic radius graph from the exact input
   geometry. Atom roles come from explicit masks or OC20 tags: 0 fixed slab,
   1 free slab, 2 adsorbate. No elements are deleted. ASE layer tags are not
   automatically OC20 tags: supply masks for structures built with ASE.
2. `SharedAdsorptionModel` embeds elements and adsorbate/slab roles, then uses
   radial message passing with a smooth quintic cutoff. Distances are recomputed
   in torch from positions, cells, and periodic image offsets.
3. Each adsorbate atom gets a configurable number of effective channels.
   Nonnegative squared couplings sum over all neighboring surface atoms/images;
   local band parameters are coupling-weighted. Independent channels are an
   approximation: they are not identified molecular orbitals and do not include
   off-diagonal orbital mixing or explicit directional orbital symmetry.
4. `SharedAdsorptionPhysics` evaluates a smoothed semi-elliptic Newns-Anderson
   model. Its energy grid, positive broadening floor, bounded state variables,
   smoothed band edges, and atan2 phase branch keep second derivatives finite.
   This deliberately differs from the head for the published checkpoints.
5. The energy is `local_energy + hybridization + orthogonalization`.
   The learned local atomic sum accommodates molecular/slab deformation,
   reference conventions, and missing interactions. These contributions are
   inspectable, but their separation is not uniquely identified by E/F labels.
6. Forces are **exact autograd derivatives of this approximate model's energy**:
   `F = -dE/dR`. Force training backpropagates through those derivatives.
   ASE rebuilds the neighbor graph after each geometry change. The current
   calculator supports fixed-cell relaxation, not stress or cell optimization.

This is a radial baseline, not an equivariant orbital Hamiltonian. Independent
effective channels and a flexible local term make a useful first ablation;
they do not establish a universal physical description. Next comparisons
should include a direct neural energy model using the same encoder, followed
by directional coupling and richer surface DOS models if held-out errors justify
them. Energy-grid convergence and state-variable bounds need validation before
production use; checkpoint results depend on the saved grid resolution.

## Install and run

Use Python 3.10+ and a current PyTorch. The implementation was exercised on CPU.

```sh
python -m pip install -e '.[dev,oc20]'
python examples/train_shared_adsorption.py --demo --epochs 60 --lr 0.003 \
  --checkpoint runs/spring-demo/model.pt

# About 2.3 MB; extracts only the S2EF training and validation LMDBs.
python examples/download_oc20_tutorial.py
python examples/train_shared_adsorption.py \
  --data data/oc20_tutorial/s2ef/train_100 \
  --validation-data data/oc20_tutorial/s2ef/val_20 \
  --energy-reference oc20-referenced --epochs 10 \
  --checkpoint runs/oc20-tutorial/model.pt
```

The tutorial source is the [official FAIR-Chem fixture](https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.0.0/tests/core/conftest.py).
The dataset is [OC20](https://doi.org/10.1021/acscatal.0c04525), licensed CC BY 4.0.
The downloader records provenance and verifies the archive's SHA256. Data and
run outputs are ignored by Git. LMDBs use pickle: load only trusted datasets.

For full OC20, point the training command at an official preprocessed S2EF
LMDB shard directory. Raw compressed trajectory archives must be preprocessed
first. The adapter supports both old and current PyG Data records, as well as
plain dictionaries. It does not depend on FAIR-Chem model packages.

## References, labels, and validation

Declare whether input energies are OC20-referenced or total. The adapter never
guesses or silently re-references labels. For raw total energies, an optional
`reference_energies` mapping subtracts the relaxed-slab-plus-gas reference for
each system exactly once. OC20's gas references are composition-based
combinations of N2, H2O, CO, and H2, not generally the isolated named adsorbate.
Forces are unchanged by a geometry-independent reference subtraction.

`sid` and `fid` are retained for split auditing. Keep every frame of a system
in one split; do not randomly split frames of relaxation trajectories. Use
official ID/OOD-adsorbate/OOD-catalyst/OOD-both partitions for scientific
evaluation. The CLI uses a separate supplied validation path and never creates
a random frame-level validation split. `y_relaxed` is not accepted as a
substitute for an S2EF frame's energy.

The loss combines energy MSE and free-atom force-component MSE. Fixed atoms
participate in the graph but are excluded from the force loss. Evaluation
reports energy MAE in eV and component force MAE in eV/Angstrom, weighted by
the actual number of structures/components. Checkpoints save architecture,
weights, precision, energy-reference convention, and training-run metadata.
They are separate from the legacy TinNet checkpoint format.

### Recorded development run (September 6, 2026)

The official tutorial contains 100 training frames from **two systems** and
20 validation frames from **one different system**, with no overlapping system
IDs. A ten-epoch CPU run (seed 0, hidden width 32, two message layers, two
channels per adsorbate atom, 4 Angstrom cutoff, 301 energy-grid points) completed
with finite energy/force gradients:

| Metric | Training | Validation |
| --- | ---: | ---: |
| Energy MAE (eV) | 1.095 | 4.752 |
| Free force-component MAE (eV/Angstrom) | 0.161 | 0.106 |

These errors are smoke-run measurements, not benchmark results or useful
accuracy claims. The next scientific step requires a much broader training
set and the official held-out evaluation splits. Local outputs are
`runs/oc20-tutorial/model.pt` and `model.metrics.json`. A three-step ASE smoke
relaxation of a validation frame returned finite forces and zero displacement
of fixed atoms; it did not reach the requested convergence tolerance.

## Predict and relax

```python
from ase.io import read
from tinnet.tinnet import SharedAdsorptionCalculator, relax_adsorption
from tinnet.tinnet.s2ef import load_shared_checkpoint

model = load_shared_checkpoint("runs/oc20-tutorial/model.pt")
atoms = read("configuration.extxyz")  # OC20 tags or supply explicit masks
atoms.calc = SharedAdsorptionCalculator(model)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

result = relax_adsorption(atoms, model, fmax=0.05, steps=200)
relaxed = result["atoms"]
print(result["converged"], result["steps"])
```

The relaxation helper operates on a copy and applies FixAtoms to tag-0 atoms,
while preserving existing constraints. When using the calculator directly with
an ASE optimizer, apply FixAtoms yourself. A successful optimizer termination
on a smoke-test checkpoint is not evidence of a correct adsorption structure.

## Verification

```sh
python -m pytest tests/test_shared_adsorption.py -q
python -m pytest -q
```

Tests cover finite-difference forces, rotations/translations, atom permutation,
periodic image shifts, variable adsorbate batching, zero contact, cutoff
continuity, force-loss gradients, independent analytic energy/force training,
reference conversion, legacy LMDB records, checkpoint round trips, and fixed
atoms during ASE relaxation. They download nothing.
