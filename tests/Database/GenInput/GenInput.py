#!/usr/bin/env python

import numpy             as np
import pandas            as pd
import pickle
import os

from ase                 import units
from ase                 import io
from scipy.signal        import hilbert
from scipy               import integrate
from scipy               import interpolate
from numpy               import linalg as LA

# substrate
pdos          = pd.read_pickle( '../Ans/Clean.pickle' )

ergy          = np.linspace(-20,15,3501)
atom_index    = 12
ads_index     = 16

ergy1         = np.array(pdos[atom_index]['energy'])
dos_d         = np.array(pdos[atom_index]['d'][0])
f1            = interpolate.interp1d(ergy1, dos_d)
dos_d         = [f1(ergy[k]) if np.min(ergy1)<ergy[k]<np.max(ergy1) else 0 for k in range(len(ergy))]
dos_d         = dos_d/integrate.cumtrapz(dos_d,ergy,axis=0)[-1]
np.save   ('dos_d.npy',dos_d)
np.savetxt('dos_d.txt',dos_d)

W             = 15
dos_sp        = [(1-(ergy[k]/W)**2)**0.5 if abs(ergy[k]) < W else 0 for k in range(len(ergy))]
dos_sp        = dos_sp/integrate.cumtrapz(dos_sp,ergy,axis=0)[-1]
np.save   ('dos_sp.npy',dos_sp)
np.savetxt('dos_sp.txt',dos_sp)

DOS_ADS       = []
d             = []

# adsorbate-substrate
for i in range(1,246):
    pdos      = pd.read_pickle( '../Ans/' + str(i).zfill(3) + '.pickle' )
    ergy2     = np.array(pdos[ads_index]['energy'])
    dos_ads   = np.array(pdos[ads_index]['s'][0])
    f2        = interpolate.interp1d(ergy2, dos_ads)
    dos_ads   = [f2(ergy[k]) if np.min(ergy2)<ergy[k]<np.max(ergy2) else 0 for k in range(len(ergy))]
    dos_ads   = dos_ads/integrate.cumtrapz(dos_ads,ergy,axis=0)[-1]
    DOS_ADS.append(dos_ads)

    #atoms     = io.read('/work/common/hxin_lab/hxin/newns_anderson/VASP/HonPt111/1-dis-ns/'+str(i+1)+'/InitialGeom.traj')
    #dis       = LA.norm(atoms[-1].position-atoms[12].position)
    #d.append(dis)
    #print(dis)

np.save   ('dos_ads.npy',DOS_ADS)

np.savetxt('dos_ads.txt',DOS_ADS)

#np.save('d.npy',d)
