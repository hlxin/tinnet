from __future__ import print_function, division

import numpy as np

from ase import io

from tinnet.regression.regression import Regression

# hyperparameters
atom_fea_len = 95
n_conv = 5
h_fea_len = 174
n_h = 2

# constants
esp = -2.693696878597913
vad2 = np.loadtxt('vad2.txt')

# images
images = io.read('OH_Images_Unrelaxed.traj', index=slice(None))

# main target
ead = np.loadtxt('ead.txt')

# indices of validation and test fold 
idx_validation_fold = 0
idx_test_fold = 2

# set up model
model = Regression(images=images,
                   main_target=ead,
                   task='test',
                   data_format='test',
                   phys_model='newns_anderson_semi',
                   batch_size=len(images),
                   idx_validation_fold=idx_validation_fold,
                   idx_test_fold=idx_test_fold,
                   
                   # hyperparameter
                   atom_fea_len=atom_fea_len,
                   n_conv=n_conv,
                   h_fea_len=h_fea_len,
                   n_h=n_h,
                   
                   # user-defined constants, additional targets, etc.
                   constant_1=esp,
                   constant_3=vad2,
                   )

train_mae, train_mse, val_mae, val_mse, test_mae, test_mse = model.predict()

print(train_mae, train_mse, val_mae, val_mse, test_mae, test_mse)
