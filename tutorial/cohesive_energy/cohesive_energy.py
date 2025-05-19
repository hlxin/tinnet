from __future__ import print_function, division

import numpy as np

from ase import io
from ase.db import connect

from cohesive_energy.tinnet.regression.regression import Regression


def cohesive_energy(reference_image=None,
                    reference_name='Reference',
                    target_image=None,
                    target_name='Target'):
    
    images = [reference_image, reference_image]
    
    # hyperparameters
    lr = 0.002
    atom_fea_len = 150
    n_conv = 5
    h_fea_len = 128
    n_h = 2
    
    pe = []
    vws = []
    
    V_dic = {'Sc': 24.724542268797304, 'Ti': 17.424646217258488, 'V': 13.917982774094817, 'Cr': 11.863283203031445, 
             'Mn': 10.785494076584158, 'Fe': 11.451083083113335, 'Co': 10.398800712477017, 'Ni': 10.916490557546439, 
             'Cu': 12.14350679198855, 'Y': 32.44397679442938, 'Zr': 23.170475005920792, 'Nb': 18.71258535775407, 
             'Mo': 16.197202890320533, 'Ru': 14.229567963011105, 'Rh': 14.229567963011105, 'Pd': 15.35668980826143, 
             'Ag': 17.969127631160283, 'Ta': 18.71258535775407, 'W': 16.54181279434536, 'Re': 14.70561133998898, 
             'Os': 14.229567963011105, 'Ir': 14.545769144994983, 'Pt': 15.857412626116277, 'Au': 18.153110451663995}
    
    PE_dic = {'Sc':2.2603424, 'Ti':2.440124875909091, 'V':2.6619930170731707, 'Cr':3.6188933459259256, 
              'Mn':4.265695346153847, 'Fe':2.9840186344444444, 'Co':1.7877074285714285, 'Ni':0.19241192499999998, 
              'Cu':0, 'Y':1.9629624225, 'Zr':1.7412473684210525, 'Nb':1.544762289473684, 'Mo':2.9535343183673466, 
              'Tc':2.330626157894737, 'Ru':1.4199372666666668, 'Rh':1.1374149333333332, 'Pd':1.03267305, 
              'Ag':0, 'Hf':3.3832650593749998, 'Ta':2.5171466666666666,  'W':3.9899073974358963, 
              'Re':3.3005202272727274, 'Os':1.9706242272727275, 'Ir':1.8323957333333332, 'Pt':0.767994218, 'Au':0}

    for image in images:
        sym = image.get_chemical_symbols()
        pe_tmp = [PE_dic[s] for s in sym]
        pe += [pe_tmp]
        vws_tmp = [V_dic[s] for s in sym]
        vws += [vws_tmp]
    
    pe = np.array(pe, dtype=object)
    vws = np.array(vws, dtype=object)
    
    index = []
    n = 0
    for p in pe:
        index += [np.arange(n, n+len(p))]
        n = n + len(p)
    
    ech_ref = []
    
    for model_idx in range(0,10):
        # set up model
        model = Regression(images=images,
                           data_format='test',
                           phys_model='cohesive_energy',
                           optim_algorithm='AdamW',
                           batch_size=1024,
                           weight_decay=0.0001,
                           idx_val_fold=model_idx,
                           idx_test_fold=model_idx,
                           convergence_epochs=1000,
                           
                           # hyperparameter
                           lr=lr,
                           atom_fea_len=atom_fea_len,
                           n_conv=n_conv,
                           h_fea_len=h_fea_len,
                           n_h=n_h,
                           
                           # user-defined constants, additional targets, etc.
                           constant_1=pe,
                           constant_2=vws,
                           constant_3=index,
                           )
        
        output, ech_ref_tmp = model.predict()
        ech_ref += [np.average(ech_ref_tmp.detach().cpu().numpy())]
    
    images = [target_image, target_image]
    
    # hyperparameters
    lr = 0.002
    atom_fea_len = 150
    n_conv = 5
    h_fea_len = 128
    n_h = 2
    
    pe = []
    vws = []
    
    V_dic = {'Sc': 24.724542268797304, 'Ti': 17.424646217258488, 'V': 13.917982774094817, 'Cr': 11.863283203031445, 
             'Mn': 10.785494076584158, 'Fe': 11.451083083113335, 'Co': 10.398800712477017, 'Ni': 10.916490557546439, 
             'Cu': 12.14350679198855, 'Y': 32.44397679442938, 'Zr': 23.170475005920792, 'Nb': 18.71258535775407, 
             'Mo': 16.197202890320533, 'Ru': 14.229567963011105, 'Rh': 14.229567963011105, 'Pd': 15.35668980826143, 
             'Ag': 17.969127631160283, 'Ta': 18.71258535775407, 'W': 16.54181279434536, 'Re': 14.70561133998898, 
             'Os': 14.229567963011105, 'Ir': 14.545769144994983, 'Pt': 15.857412626116277, 'Au': 18.153110451663995}
    
    PE_dic = {'Sc':2.2603424, 'Ti':2.440124875909091, 'V':2.6619930170731707, 'Cr':3.6188933459259256, 
              'Mn':4.265695346153847, 'Fe':2.9840186344444444, 'Co':1.7877074285714285, 'Ni':0.19241192499999998, 
              'Cu':0, 'Y':1.9629624225, 'Zr':1.7412473684210525, 'Nb':1.544762289473684, 'Mo':2.9535343183673466, 
              'Tc':2.330626157894737, 'Ru':1.4199372666666668, 'Rh':1.1374149333333332, 'Pd':1.03267305, 
              'Ag':0, 'Hf':3.3832650593749998, 'Ta':2.5171466666666666,  'W':3.9899073974358963, 
              'Re':3.3005202272727274, 'Os':1.9706242272727275, 'Ir':1.8323957333333332, 'Pt':0.767994218, 'Au':0}

    for image in images:
        sym = image.get_chemical_symbols()
        pe_tmp = [PE_dic[s] for s in sym]
        pe += [pe_tmp]
        vws_tmp = [V_dic[s] for s in sym]
        vws += [vws_tmp]
    
    pe = np.array(pe, dtype=object)
    vws = np.array(vws, dtype=object)
    
    index = []
    n = 0
    for p in pe:
        index += [np.arange(n, n+len(p))]
        n = n + len(p)
    
    ech_target = []
    
    for model_idx in range(0,10):
        # set up model
        model = Regression(images=images,
                           data_format='test',
                           phys_model='cohesive_energy',
                           optim_algorithm='AdamW',
                           batch_size=1024,
                           weight_decay=0.0001,
                           idx_val_fold=model_idx,
                           idx_test_fold=model_idx,
                           convergence_epochs=1000,
                           
                           # hyperparameter
                           lr=lr,
                           atom_fea_len=atom_fea_len,
                           n_conv=n_conv,
                           h_fea_len=h_fea_len,
                           n_h=n_h,
                           
                           # user-defined constants, additional targets, etc.
                           constant_1=pe,
                           constant_2=vws,
                           constant_3=index,
                           )
        
        output, ech_target_tmp = model.predict()
        ech_target += [np.average(ech_target_tmp.detach().cpu().numpy())]
    return np.average(ech_ref), np.average(ech_target)