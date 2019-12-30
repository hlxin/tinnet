#!/usr/bin/env python
import sys

sys.path.insert(0, '/Users/yanghuang/blueridge/cgcnn/cgcnn/optimize_and_train/script')

import os
from modify_data import CIFData
import run_cgcnn as rc
import analyze_cgcnn as ac
import modify_main as mm
import argparse

# we are not using sigopt so we will for now manually give it the parameters required.

cif_file_path = "/Users/yanghuang/catalysis/cgcnn/data/convert/"
files = [f for f in os.listdir(cif_file_path) if f.endswith('.cif')]

print(files)

data_size = len(files)
train_size, test_size = round(data_size*0.8), round(data_size*0.1)
val_size = data_size - train_size - test_size


print('----Loading data to dataset----')
dataset = CIFData(cif_file_path)

print('----Train model----')

args = argparse.Namespace(optim ='SGD',
                          atom_fea_len=64, #Number of properties used in atom feature vector
                          batch_size=30,
                          cuda=False,
                          data_options=['/Users/yanghuang/catalysis/cgcnn/data/convert/',
                                        cif_file_path],  # not really sure what this is. We already feed in the data?
                          disable_cuda=True,
                          epochs=30,
                          h_fea_len=128,  #Length of learned atom feature vector
                          lr=.01, #learning rate
                          lr_milestones=[100],
                          momentum=0.9,
                          n_conv=3, #Number of convolutional layers
                          n_h=1,  #number of hidden layers
                          print_freq=10,
                          resume='',
                          start_epoch=0,
                          task='regression',

                          test_size=test_size,
                          train_size=train_size,
                          val_size=val_size,

                          weight_decay=0.0,
                          workers=0)

print(data_size, test_size, train_size, val_size)
epochs, train_mae_errors, train_losses, val_mae_errors, val_losses, test_mae, test_loss = mm.main_cgcnn(args, dataset=dataset)
save_convergence = ac.save_convergence
save_convergence(epochs, train_mae_errors, train_losses, val_mae_errors, val_losses)
