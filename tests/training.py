from __future__ import print_function, division

import numpy as np

from ase import io

from tinnet.regression.regression import Regression

lr = 0.0020242799159685978
lamb = 0.1
atom_fea_len = 95
n_conv = 5
h_fea_len = 174
n_h = 2
Esp = -2.693696878597913

images = io.read('OH_Images_Unrelaxed.traj', index=slice(None))

energy = np.loadtxt('Energy.txt')
d_cen = np.loadtxt('d_cen.txt')
vad2 = np.loadtxt('vad2.txt')
width = np.loadtxt('width.txt')
dos_ads_1 = np.loadtxt('dos_1.txt')
dos_ads_2 = np.loadtxt('dos_2.txt')
dos_ads_3 = np.loadtxt('dos_3.txt')

final_ans_val_mae = np.zeros(10)
final_ans_val_mse = np.zeros(10)
final_ans_test_mae = np.zeros(10)
final_ans_test_mse = np.zeros(10)

for idx_test in range(2,3):
    idx_validation = 0
    
    model = Regression(images,
                       energy,
                       task='train',
                       data_format='nested',
                       phys_model='newns_anderson_semi', # for training
                       optim_algorithm='AdamW', # for training
                       batch_size=64, # for training
                       weight_decay=0.0001, # for training
                       idx_validation=idx_validation, # for training
                       idx_test=idx_test, # for training
                       lr=lr, # for architecture
                       atom_fea_len=atom_fea_len, # for architecture
                       n_conv=n_conv, # for architecture
                       h_fea_len=h_fea_len, # for architecture
                       n_h=n_h, # for architecture
                       Esp=Esp, # for TinNet
                       lamb=lamb, # for TinNet
                       d_cen=d_cen, # for TinNet
                       width=width, # for TinNet
                       vad2=vad2, # for TinNet
                       dos_ads_1=dos_ads_1, # for TinNet
                       dos_ads_2=dos_ads_2, # for TinNet
                       dos_ads_3=dos_ads_3, # for TinNet
                       emax=15, # for TinNet
                       emin=-15, # for TinNet
                       num_datapoints=3001 # for TinNet
                       )
    
    final_ans_val_mae[idx_test], \
    final_ans_val_mse[idx_test], \
    final_ans_test_mae[idx_test], \
    final_ans_test_mse[idx_test] \
        = model.train(25000)

np.savetxt('final_ans_val_MAE.txt', final_ans_val_mae)
np.savetxt('final_ans_val_MSE.txt', final_ans_val_mse)
np.savetxt('final_ans_test_MAE.txt', final_ans_test_mae)
np.savetxt('final_ans_test_MSE.txt', final_ans_test_mse)
