from __future__ import print_function, division

import numpy as np
import multiprocessing
import torch

from ase import io

from adsorption_energy_OH_top.piml.feature.voronoi import Voronoi
from adsorption_energy_OH_top.piml.regression.regression import Regression

def adsorption_energy_OH_top(reference_image=None,
                             reference_site_idx=None,
                             reference_name='Reference',
                             target_image=None,
                             target_site_idx=None,
                             target_name='Target'):
    
    images = [reference_image, reference_image]
    tabulated_site_index = np.array([reference_site_idx, reference_site_idx])
    
    elements = {'Ca': {'f': 0.1, 'vad2': 20.8 , 'eps_d':  None},
                'Sc': {'f': 0.2, 'vad2':  7.90, 'eps_d':  None},
                'Ti': {'f': 0.3, 'vad2':  4.65, 'eps_d':  1.50},
                'V' : {'f': 0.4, 'vad2':  3.15, 'eps_d':  1.06},
                'Cr': {'f': 0.5, 'vad2':  2.35, 'eps_d':  0.16},
                'Mn': {'f': 0.6, 'vad2':  1.94, 'eps_d':  0.07},
                'Fe': {'f': 0.7, 'vad2':  1.59, 'eps_d': -0.92},
                'Co': {'f': 0.8, 'vad2':  1.34, 'eps_d': -1.17},
                'Ni': {'f': 0.9, 'vad2':  1.16, 'eps_d': -1.29},
                'Cu': {'f': 1.0, 'vad2':  1.00, 'eps_d': -2.67},
                'Zn': {'f': 1.0, 'vad2':  0.46, 'eps_d':  None},
                'Sr': {'f': 0.1, 'vad2': 36.5 , 'eps_d':  None},
                'Y' : {'f': 0.2, 'vad2': 17.3 , 'eps_d':  None},
                'Zr': {'f': 0.3, 'vad2': 10.90, 'eps_d':  1.95},
                'Nb': {'f': 0.4, 'vad2':  7.73, 'eps_d':  1.41},
                'Mo': {'f': 0.5, 'vad2':  6.62, 'eps_d':  0.35},
                'Tc': {'f': 0.6, 'vad2':  4.71, 'eps_d': -0.60},
                'Ru': {'f': 0.7, 'vad2':  3.87, 'eps_d': -1.41},
                'Rh': {'f': 0.8, 'vad2':  3.32, 'eps_d': -1.73},
                'Pd': {'f': 0.9, 'vad2':  2.78, 'eps_d': -1.83},
                'Ag': {'f': 1.0, 'vad2':  2.26, 'eps_d': -4.30},
                'Cd': {'f': 1.0, 'vad2':  1.58, 'eps_d':  None},
                'Ba': {'f': 0.1, 'vad2': 41.5 , 'eps_d':  None},
                'Lu': {'f': 0.2, 'vad2': 17.1 , 'eps_d':  None},
                'Hf': {'f': 0.3, 'vad2': 11.90, 'eps_d':  2.47},
                'Ta': {'f': 0.4, 'vad2':  9.05, 'eps_d':  2.00},
                'W' : {'f': 0.5, 'vad2':  7.27, 'eps_d':  0.77},
                'Re': {'f': 0.6, 'vad2':  6.04, 'eps_d': -0.51},
                'Os': {'f': 0.7, 'vad2':  5.13, 'eps_d':  None},
                'Ir': {'f': 0.8, 'vad2':  4.45, 'eps_d': -2.11},
                'Pt': {'f': 0.9, 'vad2':  3.90, 'eps_d': -2.25},
                'Au': {'f': 1.0, 'vad2':  3.35, 'eps_d': -3.56},
                'Hg': {'f': 1.0, 'vad2':  2.64, 'eps_d':  None}}
    
    lr = 0.0018468405467806104
    lamb = 0.1
    atom_fea_len = 77
    n_conv = 6
    h_fea_len = 79
    n_h = 4
    Esp = -2.693696878597913
    
    descriptor = Voronoi(max_num_nbr=12,
                         radius=8,
                         dmin=0,
                         step=0.2,
                         dict_atom_fea=None)
    
    vad2 = []
    for idx, image in zip(tabulated_site_index, images):
        vad2 += [elements[image.get_chemical_symbols()[idx]]['vad2']]
    vad2 = np.array(vad2, dtype=np.float32)
    
    features = multiprocessing.Pool().map(descriptor.feas, images)
    
    ead_ref = []
    
    for model_idx in range(0,10):
        model = Regression(features,
                           phys_model='newns_anderson_semi',
                           optim_algorithm='AdamW',
                           weight_decay=0.0001,
                           idx_validation=model_idx,
                           idx_test=model_idx,
                           lr=lr,
                           atom_fea_len=atom_fea_len,
                           n_conv=n_conv,
                           h_fea_len=h_fea_len,
                           n_h=n_h,
                           Esp=Esp,
                           lamb=lamb,
                           vad2=vad2,
                           tabulated_site_index=tabulated_site_index,
                           emax=15,
                           emin=-15,
                           num_datapoints=3001,
                           batch_size=1024)
        
        ead_ref_tmp, parm = model.check_loss()
        ead_ref += [ead_ref_tmp[0]]
    
    images = [target_image, target_image]
    tabulated_site_index = np.array([target_site_idx, target_site_idx])
    
    elements = {'Ca': {'f': 0.1, 'vad2': 20.8 , 'eps_d':  None},
                'Sc': {'f': 0.2, 'vad2':  7.90, 'eps_d':  None},
                'Ti': {'f': 0.3, 'vad2':  4.65, 'eps_d':  1.50},
                'V' : {'f': 0.4, 'vad2':  3.15, 'eps_d':  1.06},
                'Cr': {'f': 0.5, 'vad2':  2.35, 'eps_d':  0.16},
                'Mn': {'f': 0.6, 'vad2':  1.94, 'eps_d':  0.07},
                'Fe': {'f': 0.7, 'vad2':  1.59, 'eps_d': -0.92},
                'Co': {'f': 0.8, 'vad2':  1.34, 'eps_d': -1.17},
                'Ni': {'f': 0.9, 'vad2':  1.16, 'eps_d': -1.29},
                'Cu': {'f': 1.0, 'vad2':  1.00, 'eps_d': -2.67},
                'Zn': {'f': 1.0, 'vad2':  0.46, 'eps_d':  None},
                'Sr': {'f': 0.1, 'vad2': 36.5 , 'eps_d':  None},
                'Y' : {'f': 0.2, 'vad2': 17.3 , 'eps_d':  None},
                'Zr': {'f': 0.3, 'vad2': 10.90, 'eps_d':  1.95},
                'Nb': {'f': 0.4, 'vad2':  7.73, 'eps_d':  1.41},
                'Mo': {'f': 0.5, 'vad2':  6.62, 'eps_d':  0.35},
                'Tc': {'f': 0.6, 'vad2':  4.71, 'eps_d': -0.60},
                'Ru': {'f': 0.7, 'vad2':  3.87, 'eps_d': -1.41},
                'Rh': {'f': 0.8, 'vad2':  3.32, 'eps_d': -1.73},
                'Pd': {'f': 0.9, 'vad2':  2.78, 'eps_d': -1.83},
                'Ag': {'f': 1.0, 'vad2':  2.26, 'eps_d': -4.30},
                'Cd': {'f': 1.0, 'vad2':  1.58, 'eps_d':  None},
                'Ba': {'f': 0.1, 'vad2': 41.5 , 'eps_d':  None},
                'Lu': {'f': 0.2, 'vad2': 17.1 , 'eps_d':  None},
                'Hf': {'f': 0.3, 'vad2': 11.90, 'eps_d':  2.47},
                'Ta': {'f': 0.4, 'vad2':  9.05, 'eps_d':  2.00},
                'W' : {'f': 0.5, 'vad2':  7.27, 'eps_d':  0.77},
                'Re': {'f': 0.6, 'vad2':  6.04, 'eps_d': -0.51},
                'Os': {'f': 0.7, 'vad2':  5.13, 'eps_d':  None},
                'Ir': {'f': 0.8, 'vad2':  4.45, 'eps_d': -2.11},
                'Pt': {'f': 0.9, 'vad2':  3.90, 'eps_d': -2.25},
                'Au': {'f': 1.0, 'vad2':  3.35, 'eps_d': -3.56},
                'Hg': {'f': 1.0, 'vad2':  2.64, 'eps_d':  None}}
    
    lr = 0.0018468405467806104
    lamb = 0.1
    atom_fea_len = 77
    n_conv = 6
    h_fea_len = 79
    n_h = 4
    Esp = -2.693696878597913
    
    descriptor = Voronoi(max_num_nbr=12,
                         radius=8,
                         dmin=0,
                         step=0.2,
                         dict_atom_fea=None)
    
    vad2 = []
    for idx, image in zip(tabulated_site_index, images):
        vad2 += [elements[image.get_chemical_symbols()[idx]]['vad2']]
    vad2 = np.array(vad2, dtype=np.float32)
    
    features = multiprocessing.Pool().map(descriptor.feas, images)
    
    ead_target = []
    
    for model_idx in range(0,10):
        model = Regression(features,
                           phys_model='newns_anderson_semi',
                           optim_algorithm='AdamW',
                           weight_decay=0.0001,
                           idx_validation=model_idx,
                           idx_test=model_idx,
                           lr=lr,
                           atom_fea_len=atom_fea_len,
                           n_conv=n_conv,
                           h_fea_len=h_fea_len,
                           n_h=n_h,
                           Esp=Esp,
                           lamb=lamb,
                           vad2=vad2,
                           tabulated_site_index=tabulated_site_index,
                           emax=15,
                           emin=-15,
                           num_datapoints=3001,
                           batch_size=1024)
        
        ead_target_tmp, parm = model.check_loss()
        ead_target += [ead_target_tmp[0]]
    
    return np.average(torch.stack(ead_ref).detach().cpu().numpy()), np.average(torch.stack(ead_target).detach().cpu().numpy())

