 from __future__ import print_function, division

import os

import multiprocessing
import numpy as np
import torch

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ase import io

from tinnet.regression.regression import Regression

class TrainTinNet(tune.Trainable):
    def _setup(self, config):
        
        self.path = '/Folder Path/'
        
        # hyperparameters
        self.lr = config.get('lr', 0.01)
        self.atom_fea_len = config.get('atom_fea_len', 64)
        self.n_conv = config.get('n_conv', 3)
        self.h_fea_len = config.get('h_fea_len', 128)
        self.n_h = config.get('n_h', 1)
        
        # constants
        self.root_lamb = 0.1
        self.esp = -2.693696878597913
        self.vad2 = np.loadtxt(self.path + 'vad2.txt')
        
        # images
        self.images = io.read(self.path + 'OH_Images_Unrelaxed.traj',
                              index=slice(None))
        
        # main target
        self.ead = np.loadtxt(self.path + 'ead.txt')
        
        # additional target(s)
        self.d_cen = np.loadtxt(self.path + 'd_cen.txt')
        self.half_width = np.loadtxt(self.path + 'half_width.txt')
        self.dos_ads_3sigma = np.loadtxt(self.path + 'dos_ads_3sigma.txt')
        self.dos_ads_1pi = np.loadtxt(self.path + 'dos_ads_1pi.txt')
        self.dos_ads_4sigma = np.loadtxt(self.path + 'dos_ads_4sigma.txt')
        
    def _train(self):
        final_ans_val_mae = np.zeros(10)
        final_ans_val_mse = np.zeros(10)
        final_ans_test_mae = np.zeros(10)
        final_ans_test_mse = np.zeros(10)
        
        for idx_validation_fold in range(0,10):
            idx_test_fold = idx_validation_fold
            
            self.model = Regression(images=self.images,
                                    main_target=self.ead,
                                    task='train',
                                    data_format='regular',
                                    phys_model='newns_anderson_semi',
                                    optim_algorithm='AdamW',
                                    batch_size=64,
                                    weight_decay=0.0001,
                                    idx_validation_fold=idx_validation_fold,
                                    idx_test_fold=idx_test_fold,
                                    convergence_epochs=1000,
                                    
                                    # hyperparameter
                                    lr=self.lr,
                                    atom_fea_len=self.atom_fea_len,
                                    n_conv=self.n_conv,
                                    h_fea_len=self.h_fea_len,
                                    n_h=self.n_h,
                                    
                                    # user-defined constants,
                                    # additional targets, etc.
                                    constant_1=self.esp,
                                    constant_2=self.root_lamb,
                                    constant_3=self.vad2,
                                    additional_traget_1=self.d_cen,
                                    additional_traget_2=self.half_width,
                                    additional_traget_3=self.dos_ads_3sigma,
                                    additional_traget_4=self.dos_ads_1pi,
                                    additional_traget_5=self.dos_ads_4sigma,
                                    )
            
            final_ans_val_mae[idx_validation_fold], \
            final_ans_val_mse[idx_validation_fold],\
            final_ans_test_mae[idx_validation_fold], \
            final_ans_test_mse[idx_validation_fold] \
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
                   + str(self.esp)
                   + '_'
                   + str(self.root_lamb)
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
                   + str(self.esp)
                   + '_'
                   + str(self.root_lamb)
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
                   + str(self.esp)
                   + '_'
                   + str(self.root_lamb)
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
                   + str(self.esp)
                   + '_'
                   + str(self.root_lamb)
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
    
    if torch.cuda.is_available():
        ngpu = 1
        ncpu = int(multiprocessing.cpu_count() / torch.cuda.device_count())
    else:
        ngpu = 0
        ncpu = multiprocessing.cpu_count()
    
    analysis = tune.run(
        TrainTinNet,
        scheduler=sched,
        stop={
            'mean_accuracy': 0.001,
            'training_iteration': 20,
        },
        resources_per_trial={
            'cpu': ncpu,
            'gpu': ngpu
        },
        num_samples= 5,
        checkpoint_at_end=True,
        checkpoint_freq=20,
        config={
            'lr': 
                tune.loguniform(0.0001,0.1),
            'atom_fea_len': 
                tune.sample_from(lambda spec: np.random.randint(low=16,
                                                                high=113)),
            'n_conv': 
                tune.sample_from(lambda spec: np.random.randint(low=1,
                                                                high=11)),
            'h_fea_len': 
                tune.sample_from(lambda spec: np.random.randint(low=32,
                                                                high=225)),
            'n_h': 
                tune.sample_from(lambda spec: np.random.randint(low=1,
                                                                high=11)),
        })
    
    print('Best config is:', analysis.get_best_config(metric='mean_accuracy',
                                                      mode='min'))
    