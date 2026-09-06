#!/usr/bin/env python
"""Template: extend TinNet to a new adsorbate and train it with the same framework.

Three steps, none of which touch the theory code:

1. Describe the adsorbate's frontier orbitals with an ``AdsorbateSpec``.
2. Build ``TheoryInfusedModel(gnn, NewnsAndersonModel(spec))``.
3. ``fit`` on (structure, site, DFT adsorption energy) data; save the
   checkpoints where ``spec.checkpoint`` points, and ``AdsorptionEnergy``
   can use the new adsorbate immediately.

This script uses *synthetic* targets so it runs anywhere; replace
``make_dataset`` with your DFT data.
"""

import numpy as np
import torch
from ase.build.surface import fcc111

from tinnet import tinnet
from tinnet.tinnet.adsorption_energy import AdsorptionEnergy
from tinnet.tinnet.descriptors import VoronoiGraph
from tinnet.tinnet.gnn import CrystalGraphConvNet
from tinnet.tinnet.physics import AdsorbateSpec, NewnsAndersonModel, OrbitalSpec, register_adsorbate
from tinnet.tinnet.tinnet_utils import pretrained_path
from tinnet.tinnet.training import TheoryInfusedModel, fit, save_checkpoint

# ---------------------------------------------------------------------------
# 1. Describe the adsorbate.  Atomic N atop: a p_z orbital (sigma-like, g=1)
#    and a degenerate p_xy pair (pi-like, g=2).  ``alpha`` and ``esp`` are the
#    only theory constants; everything else is learned.
# ---------------------------------------------------------------------------
N_ATOP = AdsorbateSpec(
    name="N",
    orbitals=(
        OrbitalSpec("pz", r"p_{z}", degeneracy=1, alpha=0.07),
        OrbitalSpec("pxy", r"p_{xy}", degeneracy=2, alpha=0.05),
    ),
    esp=-4.0,
    geometry=(("N", (0.0, 0.0, 1.55)),),
    d_band="latent",  # let the GNN predict d_cen / width (like OH); 'tabulated' uses BandCenter (like O)
    gnn=dict(atom_fea_len=64, n_conv=3, h_fea_len=64, n_h=2),
    checkpoint=("adsorption_energy", "N", "atop"),
    n_ensemble=2,
)
register_adsorbate(N_ATOP, overwrite=True)
physics = NewnsAndersonModel(N_ATOP)
print(physics.describe())


# ---------------------------------------------------------------------------
# 2. Data: (slab, site index) -> graph features + constants + target.
# ---------------------------------------------------------------------------
def synthetic_target(constants, rng):
    """Stand-in for a DFT adsorption energy.

    The theory module is evaluated on a hand-picked set of parameters so the
    synthetic targets are consistent with the Newns-Anderson model.  Replace
    this function with a lookup of your DFT data.
    """
    # [eps_pz, beta_pz, delta_pz, eps_pxy, beta_pxy, delta_pxy, d_cen, width] before activations
    raw = np.array([-5.0, -0.5, -1.0, -3.0, -1.0, -1.0, -2.0, 1.0]) + rng.normal(0, 0.3, 8)
    with torch.no_grad():
        energy, _ = physics(torch.tensor(raw, dtype=torch.float32)[None, :],
                            {k: torch.as_tensor(v) for k, v in constants.items()})
    return float(energy)


def make_dataset(n_structures=8, seed=0):
    rng = np.random.default_rng(seed)
    facade = AdsorptionEnergy(adsorbate=N_ATOP)
    descriptor = VoronoiGraph(max_num_nbr=12, radius=8, dmin=0, step=0.2)
    items = []
    for _ in range(n_structures):
        host = rng.choice(["Cu", "Ag", "Au", "Pd", "Pt"])
        slab = fcc111(host, size=(2, 2, 3), a=3.6 + 0.4 * rng.random())
        slab.center(vacuum=6.0, axis=2)
        slab[-1].symbol = rng.choice(["Cu", "Ni", "Pd", "Pt"])
        site = len(slab) - 1
        ads = facade.build_adsorbed_image(slab, site, N_ATOP)
        atom_fea, nbr_fea, nbr_idx = descriptor.feas(ads)
        constants = facade.site_constants(slab, [site])
        target = synthetic_target(constants, rng)  # <- replace with the DFT adsorption energy
        items.append((atom_fea, nbr_fea, nbr_idx, site, constants, target))
    return items


def collate(items):
    """Batch several crystal graphs for the shared CGCNN."""
    atom_fea, nbr_fea, nbr_idx, crystal_idx, sites, targets = [], [], [], [], [], []
    constants = {}
    offset = 0
    for a, n, i, site, c, t in items:
        atom_fea.append(torch.as_tensor(a, dtype=torch.float32))
        nbr_fea.append(torch.as_tensor(n, dtype=torch.float32))
        nbr_idx.append(torch.as_tensor(i, dtype=torch.long) + offset)
        crystal_idx.append(torch.arange(len(a), dtype=torch.long) + offset)
        sites.append(site)
        targets.append(t)
        for k, v in c.items():
            constants.setdefault(k, []).append(float(np.asarray(v).reshape(-1)[0]))
        offset += len(a)
    return {
        "inputs": (torch.cat(atom_fea), torch.cat(nbr_fea), torch.cat(nbr_idx), crystal_idx, sites),
        "constants": {k: torch.tensor(v) for k, v in constants.items()},
        "target": torch.tensor(targets, dtype=torch.float32),
    }


def main():
    items = make_dataset()
    batch = collate(items)
    atom_fea, nbr_fea = batch["inputs"][0], batch["inputs"][1]

    # ------------------------------------------------------------------
    # 3. Train an ensemble through the theory head and save checkpoints.
    # ------------------------------------------------------------------
    out_dir = pretrained_path(*N_ATOP.checkpoint)  # where AdsorptionEnergy will look for 'N'
    for member in range(N_ATOP.n_ensemble):
        torch.manual_seed(member)
        gnn = CrystalGraphConvNet(atom_fea.shape[-1], nbr_fea.shape[-1], n_out=physics.n_latent,
                                  readout="mean", **N_ATOP.gnn)
        model = TheoryInfusedModel(gnn, physics)
        history = fit(model, [batch], epochs=150, lr=3e-3, log_every=50)
        path = save_checkpoint(model, out_dir / f"model_{member}.pth.tar")
        print(f"member {member}: final loss {history['train'][-1]:.3e} -> {path}")

    # The facade now works for 'N' exactly like for 'OH' and 'O'.
    slab = fcc111("Pt", size=(2, 2, 3), a=3.92)
    slab.center(vacuum=6.0, axis=2)
    predictor = tinnet.AdsorptionEnergy(image=slab, site_inx=[len(slab) - 1], adsorbate="N", name="Pt(111)")
    predictor.predict()
    print({k: round(float(v), 3) for k, v in predictor.decompose().items()})


if __name__ == "__main__":
    main()
