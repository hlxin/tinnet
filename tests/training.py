from __future__ import print_function, division

import numpy as np

from ase import io
from ase.db import connect

from tinnet.regression.regression import Regression

# hyperparameters
lr = 0.0020242799159685978
atom_fea_len = 95
n_conv = 5
h_fea_len = 174
n_h = 2

# database
db = connect('tinnet_OH.db')

# constants
esp = -2.693696878597913
root_lamb = 0.1
vad2 = np.array([r['data']['vad2'] for r in db.select()])

# images
images = [r.toatoms() for r in db.select()]

# main target
ead = np.array([r['ead'] for r in db.select()])

# additional target(s)
d_cen = np.array([r['data']['d_cen'] for r in db.select()])
half_width = np.array([r['data']['half_width'] for r in db.select()])
dos_ads_3sigma = np.array([r['data']['dos_ads_3sigma'] for r in db.select()])
dos_ads_1pi = np.array([r['data']['dos_ads_1pi'] for r in db.select()])
dos_ads_4sigma = np.array([r['data']['dos_ads_4sigma'] for r in db.select()])

# indices of validation and test fold 
idx_val_fold = 0
idx_test_fold = 2

# set up model
model = Regression(images=images,
                   main_target=ead,
                   data_format='nested',
                   phys_model='newns_anderson_semi',
                   optim_algorithm='AdamW',
                   batch_size=64,
                   weight_decay=0.0001,
                   idx_val_fold=idx_val_fold,
                   idx_test_fold=idx_test_fold,
                   #train_ratio=1.0,
                   #val_ratio=0.0,
                   #test_ratio=0.0,
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
train_mae, train_mse, val_mae, val_mse, test_mae, test_mse = model.train(25000)

print(train_mae, train_mse, val_mae, val_mse, test_mae, test_mse)
