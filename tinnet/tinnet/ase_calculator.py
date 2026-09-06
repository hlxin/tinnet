"""ASE energy/force calculation and fixed-cell adsorption relaxation."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from .structures import configuration_graph, move_batch


class SharedAdsorptionCalculator(Calculator):
    """ASE interface; neighbors are rebuilt after motion. No stress support.

    Energy follows the checkpoint's training reference. Tags or explicit
    masks identify adsorbates. ASE, not the model, applies force constraints.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, *, adsorbate_mask=None, fixed_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model.eval()
        self.adsorbate_mask = (
            None if adsorbate_mask is None else np.array(adsorbate_mask, copy=True)
        )
        self.fixed_mask = (
            None if fixed_mask is None else np.array(fixed_mask, copy=True)
        )

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        if self.atoms is not None and not np.array_equal(
            self.atoms.get_tags(), atoms.get_tags()
        ):
            changes.append("tags")
        return changes

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        parameter = next(self.model.parameters())
        graph = configuration_graph(
            self.atoms,
            adsorbate_mask=self.adsorbate_mask,
            fixed_mask=self.fixed_mask,
            cutoff=self.model.cutoff,
            dtype=parameter.dtype,
        )
        output = self.model(move_batch(graph, parameter.device), compute_forces=True)
        self.results = {
            "energy": float(output["energy"][0].detach().cpu()),
            "forces": output["forces"].detach().cpu().numpy(),
        }


def relax_adsorption(
    atoms,
    model,
    *,
    adsorbate_mask=None,
    fixed_mask=None,
    fmax=0.05,
    steps=200,
    trajectory=None,
    logfile=None,
):
    """Relax a copy with FIRE, retaining constraints and fixing tag-0 atoms.

    Returns atoms, convergence flag, and step count. Meaningful predictions
    require trained and validated weights, not a random model or smoke run.
    """
    relaxed = atoms.copy()
    graph = configuration_graph(
        relaxed,
        adsorbate_mask=adsorbate_mask,
        fixed_mask=fixed_mask,
        cutoff=model.cutoff,
    )
    fixed = graph["fixed_mask"].numpy()
    if fixed.any():
        relaxed.set_constraint([*relaxed.constraints, FixAtoms(mask=fixed)])
    relaxed.calc = SharedAdsorptionCalculator(
        model, adsorbate_mask=adsorbate_mask, fixed_mask=fixed
    )
    optimizer = FIRE(relaxed, trajectory=trajectory, logfile=logfile)
    converged = optimizer.run(fmax=fmax, steps=steps)
    return {"atoms": relaxed, "converged": bool(converged), "steps": optimizer.nsteps}
