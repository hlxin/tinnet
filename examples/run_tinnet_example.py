#!/usr/bin/env python
"""Predict properties of a Cu single-atom alloy in Ag(111) with the pretrained models."""

from ase.build import fcc111

from tinnet import AdsorptionEnergy, BandCenter, BandMoments, CohesiveEnergy


def slab(host, a, size=(2, 2, 4), guests=()):
    image = fcc111(host, size=size, a=a)
    image.center(vacuum=7.5, axis=2)
    for i in guests:
        image[i].symbol = 'Cu'
    return image


cu_ag_saa = slab('Ag', 4.15, guests=[-1])
cu_dimer_ag = slab('Ag', 4.15, size=(4, 4, 4), guests=[-1, -2])

# Each predict() prints a summary table and returns the per-model ensemble predictions.
AdsorptionEnergy(image=cu_ag_saa, site_inx=[15], adsorbate='O', name='Cu1Ag SAA').predict()
BandCenter(image=cu_ag_saa, atom_inx=15, name='Cu1Ag SAA').predict()
BandMoments(image=cu_ag_saa, atom_inx=15, name='Cu1Ag SAA').predict()
CohesiveEnergy(image=cu_dimer_ag, name='Cu_dimer_Ag').predict()
