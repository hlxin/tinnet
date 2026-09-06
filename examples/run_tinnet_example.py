#!/usr/bin/env python
"""Reproduce the notebook examples with the pretrained TinNet models.

Run from the repository root (so that ``data/`` is found)::

    python examples/run_tinnet_example.py
"""

import pickle
from pathlib import Path

import numpy as np
from ase import io
from ase.build.surface import fcc111

from tinnet import tinnet

ROOT = Path(__file__).resolve().parents[1]


def build_structures():
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
    cu_di_saa_ag = slab("Ag", (4, 4, 4))
    cu_di_saa_ag[-1].symbol = "Cu"
    cu_di_saa_ag[-3].symbol = "Cu"
    images = io.read(ROOT / "data" / "OH_images.traj", index=slice(None))
    return dict(pure_cu=pure_cu, cu_ag_saa=cu_ag_saa, cu_dimer_ag=cu_dimer_ag,
                cu_di_saa_ag=cu_di_saa_ag, pure_pt=images[0], pt_mlpt3co=images[1])


def main():
    s = build_structures()

    # 1. Adsorption energy: any registered adsorbate, same code path.
    print("Registered adsorbates:", tinnet.available_adsorbates())
    model = tinnet.AdsorptionEnergy(image=s["pt_mlpt3co"], site_inx=[20], adsorbate="OH", name="Pt_MLPt3Co111")
    model.predict()
    print(model.physics.describe())
    terms = model.decompose()
    for orbital in model.spec.orbitals:
        print(f"  {orbital.name:8s} hybridization {terms['hyb_' + orbital.name]:+.3f} eV, "
              f"orthogonalization {terms['ortho_' + orbital.name]:+.3f} eV")
    model.explain_shap(ref_image=s["pure_pt"], ref_site_inx=[5], ref_name="Pure Pt", plot_name="shap_OH")

    model = tinnet.AdsorptionEnergy(image=s["cu_ag_saa"], site_inx=[15], adsorbate="O", name="Cu1Ag SAA")
    model.predict()

    # 2. d-band center (rectangular band theory)
    model = tinnet.BandCenter(image=s["cu_ag_saa"], atom_inx=15, name="Cu1Ag SAA")
    model.predict()

    # 3. d-band moments (tight-binding moment theory)
    model = tinnet.BandMoments(image=s["cu_ag_saa"], atom_inx=15, name="Cu1Ag SAA")
    model.predict()

    # 4. Cohesive energy (renormalized-atom cohesion theory)
    model = tinnet.CohesiveEnergy(image=s["cu_dimer_ag"], name="Cu_dimer_Ag")
    ech = model.predict()
    print("cohesive energy ensemble:", np.round(ech, 3))


if __name__ == "__main__":
    main()
