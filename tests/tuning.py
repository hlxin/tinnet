from __future__ import print_function, division

import os

import numpy as np
import torch

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ase import io

from tinnet.regression.regression import Regression

class TrainTINN(tune.Trainable):
    def _setup(self, config):
        
        self.lr = config.get('lr', 0.01)
        self.lamb = 0.1
        self.atom_fea_len = config.get('atom_fea_len', 64)
        self.n_conv = config.get('n_conv', 3)
        self.h_fea_len = config.get('h_fea_len', 128)
        self.n_h = config.get('n_h', 1)
        self.Esp = -2.693696878597913
        
        self.path = '/work/cascades/wangsh/Machine_Learning/Database/111/Training/OH/Code/Tuning_Hyperparameters/'
        
        self.images = io.read(self.path + 'OH_Images_Unrelaxed.traj',
                              index=slice(None))
        
        self.energy = np.loadtxt(self.path + 'Energy.txt')
        self.d_cen = np.loadtxt(self.path + 'd_cen.txt')
        self.vad2 = np.loadtxt(self.path + 'vad2.txt')
        self.width = np.loadtxt(self.path + 'width.txt')
        self.dos_ads_1 = np.loadtxt(self.path + 'dos_1.txt')
        self.dos_ads_2 = np.loadtxt(self.path + 'dos_2.txt')
        self.dos_ads_3 = np.loadtxt(self.path + 'dos_3.txt')
        
    def _train(self):
        final_ans_val_mae = np.zeros(10)
        final_ans_val_mse = np.zeros(10)
        final_ans_test_mae = np.zeros(10)
        final_ans_test_mse = np.zeros(10)
        
        for idx_fold in range(0,10):
            idx_test = idx_fold
            
            self.model = Regression(self.images,
                                    self.energy,
                                    task='train',
                                    data_format='regular',
                                    phys_model='newns_anderson_semi',
                                    optim_algorithm='AdamW',
                                    batch_size=64,
                                    weight_decay=0.0001,
                                    idx_validation=idx_fold,
                                    idx_test=idx_test,
                                    lr=self.lr,
                                    atom_fea_len=self.atom_fea_len,
                                    n_conv=self.n_conv,
                                    h_fea_len=self.h_fea_len,
                                    n_h=self.n_h,
                                    Esp=self.Esp,
                                    lamb=self.lamb,
                                    d_cen=self.d_cen,
                                    width=self.width,
                                    vad2=self.vad2,
                                    dos_ads_1=self.dos_ads_1,
                                    dos_ads_2=self.dos_ads_2,
                                    dos_ads_3=self.dos_ads_3,
                                    emax=15,
                                    emin=-15,
                                    num_datapoints=3001
                                    )
            
            final_ans_val_mae[idx_fold], \
            final_ans_val_mse[idx_fold],\
            final_ans_test_mae[idx_fold], \
            final_ans_test_mse[idx_fold] \
                    = self.model.train(25000)
            
            if np.max(final_ans_test_mse) > 1e5:
                break
        
        np.savetxt(self.path + 'final_ans_val_mae_'
                   + str(self.lr)
                   + '_'
                   + str(self.atom_fea_len)
                   + '_'
                   + str(self.n_conv)
                   + '_'
                   + str(self.h_fea_len)
                   + '_'
                   + str(self.n_h)
                   + '_'
                   + str(self.Esp)
                   + '_'
                   + str(self.lamb)
                   + '.txt', final_ans_val_mae)
        
        np.savetxt(self.path + 'final_ans_val_mse_'
                   + str(self.lr)
                   + '_'
                   + str(self.atom_fea_len)
                   + '_'
                   + str(self.n_conv)
                   + '_'
                   + str(self.h_fea_len)
                   + '_'
                   + str(self.n_h)
                   + '_'
                   + str(self.Esp)
                   + '_'
                   + str(self.lamb)
                   + '.txt', final_ans_val_mse)
        
        np.savetxt(self.path + 'final_ans_test_mae_'
                   + str(self.lr)
                   + '_'
                   + str(self.atom_fea_len)
                   + '_'
                   + str(self.n_conv)
                   + '_'
                   + str(self.h_fea_len)
                   + '_'
                   + str(self.n_h)
                   + '_'
                   + str(self.Esp)
                   + '_'
                   + str(self.lamb)
                   + '.txt', final_ans_test_mae)
        
        np.savetxt(self.path + 'final_ans_test_mse_'
                   + str(self.lr)
                   + '_'
                   + str(self.atom_fea_len)
                   + '_'
                   + str(self.n_conv)
                   + '_'
                   + str(self.h_fea_len)
                   + '_'
                   + str(self.n_h)
                   + '_'
                   + str(self.Esp)
                   + '_'
                   + str(self.lamb)
                   + '.txt', final_ans_test_mse)
        
        return {'mean_accuracy': np.average(final_ans_test_mse)}
    
    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
    
if __name__ == '__main__':
    
    sched = ASHAScheduler(metric='mean_accuracy')
    
    analysis = tune.run(
        TrainTINN,
        scheduler=sched,
        stop={
            'mean_accuracy': 0.001,
            'training_iteration': 20,
        },
        resources_per_trial={
            'cpu': 12,
            'gpu': 1
        },
        num_samples= 2,
        checkpoint_at_end=True,
        checkpoint_freq=20,
        config={
            'lr': tune.loguniform(0.0001,0.1),
            'atom_fea_len': tune.sample_from(lambda spec: np.random.randint(low=16, high=113)),
            'n_conv': tune.sample_from(lambda spec: np.random.randint(low=1, high=11)),
            'h_fea_len': tune.sample_from(lambda spec: np.random.randint(low=32, high=225)),
            'n_h': tune.sample_from(lambda spec: np.random.randint(low=1, high=11)),
        })
    
    print('Best config is:', analysis.get_best_config(metric='mean_accuracy',
                                                      mode='min'))
    