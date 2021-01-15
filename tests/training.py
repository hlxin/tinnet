from __future__ import print_function, division

import numpy as np

from ase import io

from tinnet.regression.regression import Regression

# hyperparameters
lr = 0.0020242799159685978
atom_fea_len = 95
n_conv = 5
h_fea_len = 174
n_h = 2

# constants
esp = -2.693696878597913
root_lamb = 0.1
vad2 = np.loadtxt('vad2.txt')

# images
images = io.read('OH_Images_Unrelaxed.traj', index=slice(None))

# main target
ead = np.loadtxt('ead.txt')

# additional target(s)
d_cen = np.loadtxt('d_cen.txt')
half_width = np.loadtxt('half_width.txt')
dos_ads_3sigma = np.loadtxt('dos_ads_3sigma.txt')
dos_ads_1pi = np.loadtxt('dos_ads_1pi.txt')
dos_ads_4sigma = np.loadtxt('dos_ads_4sigma.txt')

# indices of validation and test fold 
idx_validation_fold = 0
idx_test_fold = 2

# set up model
model = Regression(images=images,
                   main_target=ead,
                   task='train',
                   data_format='nested',
                   phys_model='newns_anderson_semi',
                   optim_algorithm='AdamW',
                   batch_size=64,
                   weight_decay=0.0001,
                   idx_validation_fold=idx_validation_fold,
                   idx_test_fold=idx_test_fold,
                   convergence_epochs=1000,
                   
                   # hyperparameter
                   lr=lr,
                   atom_fea_len=atom_fea_len,
                   n_conv=n_conv,
                   h_fea_len=h_fea_len,
                   n_h=n_h,
                   
                   # user-defined constants, additional targets, etc.
                   constant_1=esp,
                   constant_2=root_lamb,
                   constant_3=vad2,
                   additional_traget_1=d_cen,
                   additional_traget_2=half_width,
                   additional_traget_3=dos_ads_3sigma,
                   additional_traget_4=dos_ads_1pi,
                   additional_traget_5=dos_ads_4sigma,
                   )

# train a model
val_mae, val_mse, test_mae, test_mse = model.train(25000)

print(val_mae, val_mse, test_mae, test_mse)
