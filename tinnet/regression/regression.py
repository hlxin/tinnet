#!/usr/bin/env python
# This script is adapted from scripts of Jeffrey C. Grossman 
# and Zachary W. Ulissi.

from __future__ import print_function, division

import os
import time
import random
import shutil
import pickle
from copy import deepcopy

import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase.db import connect
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from tinnet.feature.voronoi import Voronoi
from tinnet.theory.theory import Chemisorption


class Regression:
    def __init__(self,
                 images,
                 main_target,
                 data_format,
                 phys_model='gcnn',
                 optim_algorithm='Adam',
                 weight_decay=0,
                 momentum=0.9,
                 batch_size=256,
                 idx_val_fold=0,
                 idx_test_fold=0,
                 train_ratio=0.9,
                 val_ratio=0.1,
                 test_ratio=0.0,
                 num_workers=0,
                 lr_milestones=[100],
                 resume=None,
                 random_seed=1234,
                 start_epoch=0,
                 convergence_epochs=1000,
                 
                 # hyperparameters
                 lr=0.001,
                 atom_fea_len=64,
                 n_conv=3,
                 h_fea_len=128,
                 n_h=1,
                 
                 # parameters for Voronoi descriptor
                 max_num_nbr=12,
                 radius=8,
                 dmin=0,
                 step=0.2,
                 dict_atom_fea=None,
                 
                 **kwargs
                 ):
        
        # initialize physical model
        Chemisorption.__init__(self, phys_model, main_target, **kwargs)
        
        # initial settings
        best_mse_error =  np.inf
        cuda = torch.cuda.is_available()
        collate_fn = self.collate_pool
        
        # calculate graph features (Voronoi descriptor)
        descriptor = Voronoi(max_num_nbr=max_num_nbr,
                             radius=radius,
                             dmin=dmin,
                             step=step,
                             dict_atom_fea=dict_atom_fea)
        
        
        try:
            features = multiprocessing.Pool().map(descriptor.feas, images)
        except:
            features = [descriptor.feas(image) for image in images]
        
        atom_fea = np.array([x[0] for x in features])
        nbr_fea = np.array([x[1] for x in features])
        nbr_fea_idx = np.array([x[2] for x in features])
        
        idx_images = np.arange(len(atom_fea))
        
        # set up dataset and loaders
        dataset = [((torch.Tensor(atom_fea[i]),
                     torch.Tensor(nbr_fea[i]),
                     torch.LongTensor(nbr_fea_idx[i])),
                    torch.Tensor([self.target[i]]),
                    idx_images[i])
                   for i in range(len(atom_fea))]
        
        train_loader, val_loader, test_loader =\
            self.get_train_val_test_loader(dataset=dataset,
                                           collate_fn=collate_fn,
                                           batch_size=batch_size,
                                           idx_val_fold=idx_val_fold,
                                           idx_test_fold=idx_test_fold,
                                           train_ratio=train_ratio,
                                           val_ratio=val_ratio,
                                           test_ratio=test_ratio,
                                           num_workers=num_workers,
                                           pin_memory=cuda,
                                           random_seed=random_seed,
                                           data_format=data_format)
        
        # build model
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len,
                                    nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    h_fea_len=h_fea_len,
                                    n_h=n_h,
                                    model_num_input=self.model_num_input)
        
        if cuda:
            model.cuda()
        
        # define loss function
        criterion = nn.MSELoss()
        
        # define optimizer
        # users can add their own optimizers as needed
        if optim_algorithm == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
        elif optim_algorithm == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr,
                                   weight_decay=weight_decay)
        elif optim_algorithm == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr,
                                    weight_decay=weight_decay)
        else:
            raise NameError('Only SGD, Adam or AdamW is allowed')
        
        scheduler = MultiStepLR(optimizer,
                                milestones=lr_milestones,
                                gamma=0.1)
        
        # optionally resume from a checkpoint
        if resume:
            if os.path.isfile(resume):
                print('=> loading checkpoint "{}"'.format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_mse_error = checkpoint['best_mse_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint "{}" (epoch {})'
                      .format(resume, checkpoint['epoch']))
            else:
                print('=> no checkpoint found at "{}"'.format(resume))
        
        self.cuda = cuda
        self.phys_model = phys_model
        self.start_epoch = start_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_mse_error = best_mse_error
        self.best_counter = 0
        self.convergence_epochs = convergence_epochs
        self.images = images
    
    def train(self, epochs=10, **kwargs):
        
        epoch_loss_pkl = open('epoch_loss.pkl', 'wb')
        epoch_loss = {}
        
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            self.epoch = epoch
            train_mse, train_mae = self.train_model(**kwargs)
            
            # evaluate on validation set
            val_mse, val_mae \
                = self.eval_model('validation', self.val_loader, **kwargs)
            
            # evaluate on test set
            test_mse, test_mae \
                = self.eval_model('test', self.test_loader, **kwargs)
            
            epoch_loss[epoch] = {}
            epoch_loss[epoch]['train_mse'] = train_mse.item()
            epoch_loss[epoch]['train_mae'] = train_mae.item()
            epoch_loss[epoch]['val_mse'] = val_mse.item()
            epoch_loss[epoch]['val_mae'] = val_mae.item()
            epoch_loss[epoch]['test_mse'] = test_mse.item()
            epoch_loss[epoch]['test_mae'] = test_mae.item()
            
            if val_mse != val_mse:
                print('Exit due to NaN')
                pickle.dump(epoch_loss, epoch_loss_pkl)
                epoch_loss_pkl.close()
                return None, None, None, None, None, None
            
            self.scheduler.step()
            
            # remember the best mse_eror and save checkpoint
            is_best = val_mse < self.best_mse_error
            self.best_mse_error = min(val_mse, self.best_mse_error)
            
            self.save_checkpoint({'epoch': epoch + 1,
                                  'state_dict': self.model.state_dict(),
                                  'best_mse_error': self.best_mse_error,
                                  'optimizer': self.optimizer.state_dict(),
                                  }, is_best, **kwargs)

            if self.best_counter >= self.convergence_epochs:
                print('Exit due to converged')
                shutil.copyfile('model_best.pth.tar',
                                'model_best_converged.pth.tar')
                os.remove('model_best.pth.tar')
                pickle.dump(epoch_loss, epoch_loss_pkl)
                epoch_loss_pkl.close()
                return self.predict()
        
        pickle.dump(epoch_loss, epoch_loss_pkl)
        epoch_loss_pkl.close()
        
        return None, None, None, None, None, None

    def predict(self, **kwargs):
        # test best model
        best_checkpoint = torch.load('model_best_converged.pth.tar')
        
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        if os.path.exists('tinnet_output.db'):
            os.remove('tinnet_output.db')
        
        train_mse, train_mae \
            = self.eval_model(catagory='train',
                              data_loader=self.train_loader,
                              save_outputs=True,
                              **kwargs)
        val_mse, val_mae \
            = self.eval_model(catagory='validation',
                              data_loader=self.val_loader,
                              save_outputs=True,
                              **kwargs)
        test_mse, test_mae \
            = self.eval_model(catagory='test',
                              data_loader=self.test_loader,
                              save_outputs=True,
                              **kwargs)
        
        return train_mae, train_mse, val_mae, val_mse, test_mae, test_mse
    
    def train_model(self, **kwargs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        
        # switch to train mode
        self.model.train()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
    
            if self.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True)
                              for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
            
            if self.cuda:
                target_var = Variable(target.cuda(non_blocking=True))
            else:
                target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gcnn':
                output, parm = Chemisorption.gcnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    dos_source='dft',
                    target=target_var,
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            if self.phys_model =='user_defined':
                output, parm = Chemisorption.user_defined(self,
                                                  cnn_output,
                                                  **kwargs)
            
            loss = self.criterion(output, target_var)*output.shape[-1]
            
            # measure accuracy and record loss
            mae_error = self.mae(output.data,target_var)*output.shape[-1]
            losses.update(loss.data, target.size(0))
            mae_errors.update(mae_error, target.size(0))
            
            # compute gradient and do SGD/Adam step
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'
                  .format(self.epoch, i, len(self.train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          mae_errors=mae_errors))
        
        return losses.avg, mae_errors.avg
    
    def eval_model(self, catagory, data_loader, save_outputs=False, **kwargs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        
        # switch to evaluate mode
        self.model.eval()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(data_loader):
            with torch.no_grad():
                if self.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True)
                                  for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
                
                if self.cuda:
                    target_var = Variable(target.cuda(non_blocking=True))
                else:
                    target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gcnn':
                output, parm = Chemisorption.gcnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    dos_source='model',
                    target=target_var,
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            if self.phys_model =='user_defined':
                output, parm = Chemisorption.user_defined(self,
                                                  cnn_output,
                                                  **kwargs)
            
            loss = self.criterion(output, target_var)*output.shape[-1]
            
            # measure accuracy and record loss
            mae_error = self.mae(output.data,target_var)*output.shape[-1]
            losses.update(loss.data, target.size(0))
            mae_errors.update(mae_error, target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            print(catagory + ': [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'
                  .format(i, len(data_loader),
                          batch_time=batch_time,
                          loss=losses,
                          mae_errors=mae_errors))
            
            if save_outputs:
                with connect('tinnet_output.db') as db:
                    for i in range(len(parm)):
                        p = parm[i]
                        db.write(self.images[int(p[0].item())],
                                 catagory = catagory,
                                 index = p[0].item(),
                                 dft_ead = p[1].item(),
                                 dft_d_cen = p[2].item(),
                                 dft_half_width = p[3].item(),
                                 ed_hybridization = p[4].item(),
                                 ed_repusion = p[5].item(),
                                 model_energy = p[6].item(),
                                 model_d_cen = p[7].item(),
                                 model_half_width = p[8].item(),
                                 adse_1 = p[9].item(),
                                 beta_1 = p[10].item(),
                                 delta_1 = p[11].item(),
                                 adse_2 = p[12].item(),
                                 beta_2 = p[13].item(),
                                 delta_2 = p[14].item(),
                                 adse_3 = p[15].item(),
                                 beta_3 = p[16].item(),
                                 delta_3 = p[17].item(),
                                 alpha = p[18].item(),
                                 filling = p[19].item(),
                                 vad2 = p[20].item(),
                                 esp = self.esp)
        
        return losses.avg, mae_errors.avg
    
    def mae(self, prediction, target):
        '''
        Computes the mean absolute error between prediction and target
    
        Parameters
        ----------
    
        prediction: torch.Tensor (N, 1)
        target: torch.Tensor (N, 1)
        '''
        return torch.mean(torch.abs(target - prediction))
    
    def save_checkpoint(self, state, is_best, **kwargs):
        self.best_counter += 1
        torch.save(state, 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('checkpoint.pth.tar', 'model_best.pth.tar')
            self.best_counter = 0
    
    def get_train_val_test_loader(self,
                                  dataset,
                                  idx_val_fold=0,
                                  idx_test_fold=None,
                                  train_ratio=0.9,
                                  val_ratio=0.1,
                                  test_ratio=0.0,
                                  collate_fn=default_collate,
                                  batch_size=256,
                                  num_workers=0,
                                  pin_memory=False,
                                  random_seed=None,
                                  data_format=None):
        '''
        Utility function for dividing a dataset to train, val, test datasets.
    
        The dataset needs to be shuffled before using the function
    
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
          The full dataset to be divided.
        batch_size: int
        train_ratio: float
        val_ratio: float
        test_ratio: float
        num_workers: int
        pin_memory: bool
    
        Returns
        -------
        train_loader: torch.utils.data.DataLoader
          DataLoader that random samples the training data.
        val_loader: torch.utils.data.DataLoader
          DataLoader that random samples the validation data.
        test_loader: torch.utils.data.DataLoader
          DataLoader that random samples the test data.
        '''
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) <= 1e-6
        
        assert (0.0 <= train_ratio <= 1.0
                and 0.0 <= val_ratio <= 1.0
                and 0.0 <= test_ratio <= 1.0)
        
        indices = np.arange(len(dataset))
        n = len(indices)

        if random_seed:
            random.Random(random_seed).shuffle(indices)
        else:
            random.shuffle(indices)
            
        if data_format == 'nested':
            
            kfold = np.array_split(indices,10)
            
            kfold_val = deepcopy(kfold[idx_val_fold])
            
            try:
                kfold_test = deepcopy(kfold[idx_test_fold])
            except:
                kfold_test = []
            
            kfold_train = deepcopy([kfold[i]
                                    for i in range(0,10)
                                    if i != idx_val_fold 
                                       and i != idx_test_fold])
            
            kfold_train = np.array([item for sl in kfold_train for item in sl])
            
        elif data_format == 'regular':
            
            kfold = np.array_split(indices,10)
                
            try:
                kfold_test = deepcopy(kfold[idx_test_fold])
                kfold_rest = deepcopy([kfold[i]
                                       for i in range(0,10)
                                       if i != idx_test_fold])
                kfold_rest = np.array([item for s in kfold_rest for item in s])
                kfold = np.array_split(kfold_rest,10)
            
            except:
                kfold_test = []
            
            kfold_val = deepcopy(kfold[idx_val_fold])
            
            kfold_train = deepcopy([kfold[i]
                                    for i in range(0,10)
                                    if i != idx_val_fold])
            
            kfold_train = np.array([item for sl in kfold_train for item in sl])
            
        elif data_format == 'random':
            section_1 = train_ratio
            section_2 = section_1 + val_ratio
            section_3 = section_2 + test_ratio
            kfold_train, kfold_val, kfold_test, rest = \
                np.array_split(indices,[int(n*section_1),
                                        int(n*section_2),
                                        int(n*section_3)])
            kfold_train = np.concatenate((kfold_train,rest))

        elif data_format == 'test':
            kfold_train = []
            kfold_val = []
            kfold_test = deepcopy(indices)
            
        else:
            raise NameError('Only nested, regular, random or test is allowed')
        
        val_sampler = SubsetRandomSampler(deepcopy(kfold_val))
        test_sampler = SubsetRandomSampler(deepcopy(kfold_test))
        train_sampler = SubsetRandomSampler(deepcopy(kfold_train))
        
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  pin_memory=pin_memory)
        
        val_loader = DataLoader(dataset, batch_size=len(dataset),
                                sampler=val_sampler,
                                num_workers=num_workers,
                                collate_fn=collate_fn,
                                pin_memory=pin_memory)
        
        test_loader = DataLoader(dataset, batch_size=len(dataset),
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)
        
        return train_loader, val_loader, test_loader

    def collate_pool(self, dataset_list):
        '''
        Collate a list of data and return a batch for predicting crystal
        properties.
    
        Parameters
        ----------
    
        dataset_list: list of tuples for each data point.
          (atom_fea, nbr_fea, nbr_fea_idx, target)
    
          atom_fea: torch.Tensor shape (n_i, atom_fea_len)
          nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
          nbr_fea_idx: torch.LongTensor shape (n_i, M)
          target: torch.Tensor shape (1, )
          cif_id: str or int
    
        Returns
        -------
        N = sum(n_i); N0 = sum(i)
    
        batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type
        batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        batch_nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        target: torch.Tensor shape (N, 1)
          Target value for prediction
        batch_cif_ids: list
        '''
        batch_atom_fea = []
        batch_nbr_fea = []
        batch_nbr_fea_idx = []
        crystal_atom_idx = []
        batch_target = []
        batch_cif_ids = []
        base_idx = 0
        
        for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
                in enumerate(dataset_list):
            n_i = atom_fea.shape[0]  # number of atoms for this crystal
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
            new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
            crystal_atom_idx.append(new_idx)
            batch_target.append(target)
            batch_cif_ids.append(cif_id)
            base_idx += n_i
        return (torch.cat(batch_atom_fea, dim=0),
                torch.cat(batch_nbr_fea, dim=0),
                torch.cat(batch_nbr_fea_idx, dim=0),
                crystal_atom_idx),\
            torch.stack(batch_target, dim=0),\
            batch_cif_ids


class AverageMeter:
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConvLayer(nn.Module):
    '''
    Convolutional operation on graphs
    '''
    def __init__(self, atom_fea_len, nbr_fea_len):
        '''
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        '''
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()
    
    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        '''
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        '''
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    '''
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    '''
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 model_num_input=1):
        '''
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        '''
        super(CrystalGraphConvNet, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len, model_num_input)
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        '''
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        '''
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        out = self.fc_out(crys_fea)
        
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        '''
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        '''
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        
        return torch.cat(summed_fea, dim=0)
