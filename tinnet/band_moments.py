#!/usr/bin/env python
# This script is adapted from Xie's and Ulissi's scripts.

import pickle
import torch
import numpy as np
import torch.nn as nn

from ase import Atom
from copy import deepcopy
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io.ase import AseAtomsAdaptor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BandMoments:
    def __init__(self,
                 image=None,
                 atom_inx=None,
                 name='Name'):
        data = Features.material_dict(self)
        atom_fea_dict = Features.dict_atom_fea_default(self)
        
        self.data = data
        self.atom_fea_dict = atom_fea_dict
        self.image = image
        self.atom_inx = atom_inx
        self.name = name
    
    def predict(self,
                image=None,
                return_all_parm=False):
        input_image = self.image
        atom_inx = self.atom_inx
        system_name = self.name
        
        images = [input_image, input_image]
        atom_inx = np.array([atom_inx, atom_inx])
        
        lr = 0.0044485033567158005
        atom_fea_len = 106
        n_conv = 9
        h_fea_len = 60
        n_h = 2
        
        atom_fea_dict = self.atom_fea_dict
        data = self.data
        
        ads_images = []
        tabulated_site_element = []
        tabulated_d_cen_inf = []
        tabulated_full_width_inf = []
        tabulated_mulliken = []
        tabulated_nbr_element = []
    
        atom_fea = [] # atomic features of each site
        nbr_fea_idx = [] # index of neighboring atoms
        nbr_fea_idx_original = [] # original index of neighboring atoms
        nbr_fea = [] # bond features
        padding_filter = [] # neighoring atoms within 1 rc (3.6 A)
    
        hopping_distance = []
        hopping = []
        power_ss = []
        power_ds = []
        power_dd = []
        power_gamma_ds = []
        power_gamma_dd = []
    
        for image_idx, (s_idx, image) in enumerate(zip(atom_inx, images)):
            
            o_indices = [i for i, atom in enumerate(image) if atom.symbol == 'O']
            for i in sorted(o_indices, reverse=True):
                del image[i]
            h_indices = [i for i, atom in enumerate(image) if atom.symbol == 'H']
            for i in sorted(h_indices, reverse=True):
                del image[i]
            
            radius = 8.0
            dmin = 0.0
            step = 0.2
            
            assert dmin < radius
            assert radius - dmin > step
            
            bond_feature_filter = np.arange(dmin, radius + step, step)
            
            pos = image.get_positions()[s_idx] + [0.0, 0.0, 2.0]
            ads_image = image.copy()
            ads_image.append(Atom('H', position = pos))
            ads_images += [ads_image]
            
            n_atoms = len(image)
            
            enlarged_image = image.copy()
            
            repeat_times = 11 # It should be an odd number
            
            assert np.mod(repeat_times, 2) == 1
            
            enlarged_image = enlarged_image.repeat((repeat_times, repeat_times, 1))
            
            idx_start = int((repeat_times**2.0-1) / 2.0 * n_atoms)
            idx_end = int(idx_start + n_atoms)
            enlarged_s_idx = int(idx_start + s_idx)
            
            # Features
            atom_fea_tmp = []
            nbr_fea_idx_tmp = []
            nbr_fea_idx_original_tmp = []
            nbr_fea_tmp = []
            padding_filter_tmp = []
            
            for current_site in range(idx_start, idx_end):
                nbr_dis = enlarged_image.get_distances(current_site,
                                                       np.arange(len(enlarged_image)))
                
                nbr_lists = np.where(nbr_dis <= 3.6 * 2.0)[0] # 2 * rc
                
                nbr_dis = nbr_dis[nbr_lists]
                nbr_lists = nbr_lists[np.argsort(nbr_dis)]
                nbr_dis = np.sort(nbr_dis)
                
                idx_number = enlarged_image[current_site].number
                
                pad_size = 132
                
                assert len(nbr_lists) <= pad_size
                
                if current_site == enlarged_s_idx:
                    
                    idx_sym = enlarged_image[current_site].symbol
                    nbr_sym = np.array([enlarged_image[nbr_list].symbol
                                        for nbr_list in nbr_lists])
                    
                    tabulated_site_element += [idx_sym]
                    tabulated_d_cen_inf += [data[idx_sym]['d_cen']]
                    tabulated_full_width_inf += [data[idx_sym]['full_width']]
                    
                    idx_mulliken = (data[idx_sym][b'IonizationPotential']
                                    + data[idx_sym][b'ElectronAffinity']) / 2.0
                    nbr_mulliken = []
                    
                    for dis, sym in zip(nbr_dis, nbr_sym):
                        if dis > 0.0 and dis <= 3.6:
                            nbr_mulliken += [(data[sym][b'IonizationPotential']
                                              + data[sym][b'ElectronAffinity']) / 2.0]
                    
                    tabulated_mulliken += [idx_mulliken
                                           - np.prod(nbr_mulliken)
                                           **(1/len(nbr_mulliken))]
                    
                    tabulated_nbr_element += [np.pad(nbr_sym,
                                                     [0, pad_size-len(nbr_sym)],
                                                     mode='constant',
                                                     constant_values='XX')]
                    
                    # TB hoppings
                    # nbr_lists = site + its neighbors
                    # orb = ['s','dxy','dyz','dxz','dz2','dx2-z2'] = 6 orbitals
                    
                    ref_hopping = np.zeros((pad_size, 6, pad_size, 6))
                    ref_hopping_distance = np.zeros((pad_size, pad_size)) + 1000000
                    ref_power_ss = np.zeros((pad_size, 6, pad_size, 6))
                    ref_power_ds = np.zeros((pad_size, 6, pad_size, 6))
                    ref_power_dd = np.zeros((pad_size, 6, pad_size, 6))
                    ref_power_gamma_ds = np.zeros((pad_size, 6, pad_size, 6))
                    ref_power_gamma_dd = np.zeros((pad_size, 6, pad_size, 6))
                    
                    all_sym = enlarged_image.get_chemical_symbols()
                    all_dij = enlarged_image.get_all_distances()
                    all_rd = [data[sym]['rd'] for sym in all_sym]
                    all_d_cen = [data[sym]['d_cen'] for sym in all_sym]
                    all_pos = enlarged_image.get_positions()
                    all_r = all_pos[:,None,:] - all_pos[None,:,:]
                    
                    for ii, idx_1 in enumerate(nbr_lists):
                        for jj, idx_2 in enumerate(nbr_lists):
                            dij = all_dij[idx_1][idx_2]
                            if dij > 0.0 and dij <= 3.6 and ii != jj:
                                rd_1 = all_rd[idx_1]
                                rd_2 = all_rd[idx_2]
                                dij = all_dij[idx_1][idx_2]
                                r = all_r[idx_1][idx_2]
                                l, m, n = r/np.linalg.norm(r)
                                
                                v_ss_sigma = 7.62*(-1.4)*(1/dij**2)
                                v_sd_sigma = 7.62*(-3.16)*(rd_2**1.5/dij**3.5)
                                v_ds_sigma = 7.62*(-3.16)*(rd_1**1.5/dij**3.5)
                                
                                v_dd_sigma = 7.62*(-16.2)*(rd_1**1.5*rd_2**1.5/dij**5)
                                v_dd_pi = 7.62*(8.75)*(rd_1**1.5*rd_2**1.5/dij**5)
                                v_dd_delta = 0
                                
                                ref_hopping_distance[ii][jj] = dij
                                
                                ref_hopping[ii][0][jj][0] = v_ss_sigma
                                
                                ref_hopping[ii][0][jj][1] = 3**0.5*l*m*v_sd_sigma
                                ref_hopping[ii][0][jj][2] = 3**0.5*m*n*v_sd_sigma
                                ref_hopping[ii][0][jj][3] = 3**0.5*l*n*v_sd_sigma
                                ref_hopping[ii][0][jj][4] = (n**2 - (l**2 + m**2)/2)*v_sd_sigma
                                ref_hopping[ii][0][jj][5] = 3**0.5/2*(l**2-m**2)*v_sd_sigma
                                
                                ref_hopping[ii][1][jj][0] = 3**0.5*l*m*v_ds_sigma
                                ref_hopping[ii][2][jj][0] = 3**0.5*m*n*v_ds_sigma
                                ref_hopping[ii][3][jj][0] = 3**0.5*l*n*v_ds_sigma
                                ref_hopping[ii][4][jj][0] = (n**2 - (l**2 + m**2)/2)*v_ds_sigma
                                ref_hopping[ii][5][jj][0] = 3**0.5/2*(l**2-m**2)*v_ds_sigma
                                
                                ref_hopping[ii][1][jj][1] = 3*l**2*m**2*v_dd_sigma + (l**2 + m**2 - 4*l**2*m**2)*v_dd_pi + (n**2 + l**2*m**2)*v_dd_delta
                                ref_hopping[ii][1][jj][2] = ref_hopping[ii][2][jj][1] = l*n*(3*m**2*v_dd_sigma + (1 - 4*m**2)*v_dd_pi + (m**2 - 1)*v_dd_delta)
                                ref_hopping[ii][1][jj][3] = ref_hopping[ii][3][jj][1] = m*n*(3*l**2*v_dd_sigma + (1 - 4*l**2)*v_dd_pi + (l**2 - 1)*v_dd_delta)
                                ref_hopping[ii][1][jj][4] = ref_hopping[ii][4][jj][1] = 3**0.5*l*m*((n**2 - 0.5*(l**2 + m**2))*v_dd_sigma - 2*n**2*v_dd_pi + 0.5*(1 + n**2)*v_dd_delta)
                                ref_hopping[ii][1][jj][5] = ref_hopping[ii][5][jj][1] = l*m*(l**2 - m**2)*(1.5*v_dd_sigma - 2*v_dd_pi + 0.5*v_dd_delta)
                                
                                ref_hopping[ii][2][jj][2] = 3*n**2*m**2*v_dd_sigma + (n**2 + m**2 - 4*n**2*m**2)*v_dd_pi + (l**2 + n**2*m**2)*v_dd_delta
                                ref_hopping[ii][2][jj][3] = ref_hopping[ii][3][jj][2] = m*l*(3*n**2*v_dd_sigma + (1 - 4*n**2)*v_dd_pi + (n**2 - 1)*v_dd_delta)
                                ref_hopping[ii][2][jj][4] = ref_hopping[ii][4][jj][2] = 3**0.5*m*n*((n**2 - 0.5*(l**2 + m**2))*v_dd_sigma + (l**2 + m**2 - n**2)*v_dd_pi - 0.5*(l**2 + m**2)*v_dd_delta)
                                ref_hopping[ii][2][jj][5] = ref_hopping[ii][5][jj][2] = m*n*(1.5*(l**2 - m**2)*v_dd_sigma - (1 + 2*(l**2 - m**2))*v_dd_pi + (1 + 0.5*(l**2 - m**2))*v_dd_delta)
                                
                                ref_hopping[ii][3][jj][3] = 3*l**2*n**2*v_dd_sigma + (l**2 + n**2 - 4*l**2*n**2)*v_dd_pi + (m**2 + l**2*n**2)*v_dd_delta
                                ref_hopping[ii][3][jj][4] = ref_hopping[ii][4][jj][3] = 3**0.5*l*n*((n**2 - 0.5*(l**2 + m**2))*v_dd_sigma + (l**2 + m**2 - n**2)*v_dd_pi - 0.5*(l**2 + m**2)*v_dd_delta)
                                ref_hopping[ii][3][jj][5] = ref_hopping[ii][5][jj][3] = n*l*(1.5*(l**2 - m**2)*v_dd_sigma + (1 - 2*(l**2 - m**2))*v_dd_pi - (1 - 0.5*(l**2 - m**2))*v_dd_delta)
                                
                                ref_hopping[ii][4][jj][4] = (n**2 - 0.5*(l**2 + m**2))**2*v_dd_sigma + 3*n**2*(l**2 + m**2)*v_dd_pi + 0.75*(l**2 + m**2)**2*v_dd_delta
                                ref_hopping[ii][4][jj][5] = ref_hopping[ii][5][jj][4] = 3**0.5*(l**2 - m**2)*(0.5*(n**2 - 0.5*(l**2 + m**2))*v_dd_sigma - n**2*v_dd_pi + 0.25*(1 + n**2)*v_dd_delta)
                                
                                ref_hopping[ii][5][jj][5] = 0.75*(l**2 - m**2)**2*v_dd_sigma + (l**2 + m**2 - (l**2 - m**2)**2)*v_dd_pi + (n**2 + 0.25*(l**2 - m**2)**2)*v_dd_delta
                                
                                #######################################################
                                
                                ref_power_ss[ii][0][jj][0] = 1
                                
                                ref_power_ds[ii][0][jj][1] = 1
                                ref_power_ds[ii][0][jj][2] = 1
                                ref_power_ds[ii][0][jj][3] = 1
                                ref_power_ds[ii][0][jj][4] = 1
                                ref_power_ds[ii][0][jj][5] = 1
                                
                                ref_power_ds[ii][1][jj][0] = 1
                                ref_power_ds[ii][2][jj][0] = 1
                                ref_power_ds[ii][3][jj][0] = 1
                                ref_power_ds[ii][4][jj][0] = 1
                                ref_power_ds[ii][5][jj][0] = 1
                                
                                ref_power_dd[ii][1][jj][1] = 1
                                ref_power_dd[ii][1][jj][2] = ref_power_dd[ii][2][jj][1] = 1
                                ref_power_dd[ii][1][jj][3] = ref_power_dd[ii][3][jj][1] = 1
                                ref_power_dd[ii][1][jj][4] = ref_power_dd[ii][4][jj][1] = 1
                                ref_power_dd[ii][1][jj][5] = ref_power_dd[ii][5][jj][1] = 1
                                
                                ref_power_dd[ii][2][jj][2] = 1
                                ref_power_dd[ii][2][jj][3] = ref_power_dd[ii][3][jj][2] = 1
                                ref_power_dd[ii][2][jj][4] = ref_power_dd[ii][4][jj][2] = 1
                                ref_power_dd[ii][2][jj][5] = ref_power_dd[ii][5][jj][2] = 1
                                
                                ref_power_dd[ii][3][jj][3] = 1
                                ref_power_dd[ii][3][jj][4] = ref_power_dd[ii][4][jj][3] = 1
                                ref_power_dd[ii][3][jj][5] = ref_power_dd[ii][5][jj][3] = 1
                                
                                ref_power_dd[ii][4][jj][4] = 1
                                ref_power_dd[ii][4][jj][5] = ref_power_dd[ii][5][jj][4] = 1
                                
                                ref_power_dd[ii][5][jj][5] = 1
    
                            if ii == jj:
                                ref_hopping[ii][0][jj][0] = 0 - all_d_cen[s_idx]
                                ref_hopping[ii][1][jj][1] = all_d_cen[idx_1] - all_d_cen[s_idx]
                                ref_hopping[ii][2][jj][2] = all_d_cen[idx_1] - all_d_cen[s_idx]
                                ref_hopping[ii][3][jj][3] = all_d_cen[idx_1] - all_d_cen[s_idx]
                                ref_hopping[ii][4][jj][4] = all_d_cen[idx_1] - all_d_cen[s_idx]
                                ref_hopping[ii][5][jj][5] = all_d_cen[idx_1] - all_d_cen[s_idx]
                                
                                #######################################################
                                
                                ref_power_gamma_ds[ii][0][jj][0] = 1
                                ref_power_gamma_dd[ii][1][jj][1] = 1
                                ref_power_gamma_dd[ii][2][jj][2] = 1
                                ref_power_gamma_dd[ii][3][jj][3] = 1
                                ref_power_gamma_dd[ii][4][jj][4] = 1
                                ref_power_gamma_dd[ii][5][jj][5] = 1
                    
                    ref_hopping_distance = np.exp(-(ref_hopping_distance[..., np.newaxis] - bond_feature_filter)**2 / step**2) * (ref_hopping_distance[:,:,None] != 1000000)
                    
                    hopping_distance += [torch.tensor(ref_hopping_distance).to_sparse_coo()]
                    hopping += [torch.tensor(ref_hopping).to_sparse_coo()]
                    power_ss += [torch.tensor(ref_power_ss).to_sparse_coo()]
                    power_ds += [torch.tensor(ref_power_ds).to_sparse_coo()]
                    power_dd += [torch.tensor(ref_power_dd).to_sparse_coo()]
                    power_gamma_ds += [torch.tensor(ref_power_gamma_ds).to_sparse_coo()]
                    power_gamma_dd += [torch.tensor(ref_power_gamma_dd).to_sparse_coo()]
                
                bond_filter = (nbr_dis > 0.0) * (nbr_dis <= 3.6)
                bond_filter = np.pad(bond_filter,
                                     [0, pad_size-len(bond_filter)],
                                     mode='constant',
                                     constant_values=0)
                
                padding_filter_tmp += [bond_filter]
                
                atom_fea_tmp += [atom_fea_dict[idx_number]]
                
                nbr_fea_idx_tmp += [np.pad(np.mod(nbr_lists, n_atoms),
                                           [0, pad_size-len(nbr_lists)],
                                           mode='constant',
                                           constant_values=0)]
                
                nbr_fea_idx_original_tmp += [np.pad(nbr_lists,
                                                    [0, pad_size-len(nbr_lists)],
                                                    mode='constant',
                                                    constant_values=1000000)]
                
                bond_fea = np.exp(-(nbr_dis[..., np.newaxis]
                                    - bond_feature_filter)**2
                                  / step**2)
                
                nbr_fea_tmp += [np.pad(bond_fea,
                                       [(0, pad_size-len(bond_fea)), (0, 0)],
                                       mode='constant',
                                       constant_values=0) * bond_filter[:,None]]
            
            atom_fea += [torch.tensor(np.array(atom_fea_tmp)).to_sparse_coo()] # atomic features of each site
            nbr_fea_idx += [torch.tensor(np.array(nbr_fea_idx_tmp)).to_sparse_coo()] # index of neighboring atoms
            nbr_fea_idx_original += [np.array(nbr_fea_idx_original_tmp)] # original index of neighboring atoms
            nbr_fea += [torch.tensor(np.array(nbr_fea_tmp)).to_sparse_coo()] # bond features
            padding_filter += [torch.tensor(np.array(padding_filter_tmp)).to_sparse_coo()] # neighoring atoms within 1 rc (3.6 A)
        
        tabulated_hopping_distance = hopping_distance
        tabulated_hopping = hopping
        tabulated_power_ss = power_ss
        tabulated_power_ds = power_ds
        tabulated_power_dd = power_dd
        tabulated_power_gamma_ds = power_gamma_ds
        tabulated_power_gamma_dd = power_gamma_dd
        
        ans_image2dmoment = []
        
        for model_inx in range(0,10):
            model = Regression(atom_fea,
                               nbr_fea,
                               nbr_fea_idx,
                               phys_model='moment',
                               optim_algorithm='AdamW',
                               weight_decay=0.0001,
                               model_inx=model_inx,
                               lr=lr,
                               atom_fea_len=atom_fea_len,
                               n_conv=n_conv,
                               h_fea_len=h_fea_len,
                               n_h=n_h,
                               tabulated_site_index=atom_inx,
                               padding_filter=padding_filter,
                               tabulated_hopping_distance=tabulated_hopping_distance,
                               tabulated_hopping=tabulated_hopping,
                               tabulated_power_ss=tabulated_power_ss,
                               tabulated_power_ds=tabulated_power_ds,
                               tabulated_power_dd=tabulated_power_dd,
                               tabulated_power_gamma_ds=tabulated_power_gamma_ds,
                               tabulated_power_gamma_dd=tabulated_power_gamma_dd,
                               batch_size=16)
            
            parm = model.check_loss()
            ans_image2dmoment += [parm[0].detach().cpu().numpy()]
        
        mean_moments = np.mean(ans_image2dmoment, axis=0)
        std_moments = np.std(ans_image2dmoment, axis=0)
        
        second_mean, third_mean, fourth_mean = mean_moments
        second_std, third_std, fourth_std = std_moments
        
        print(f"--- Moments for atom index {atom_inx[0]} of {system_name} ---")
        if second_mean is not None:
            print(f"Second Moment  (⟨ε²⟩): {second_mean:.2f} ± {second_std:.2f}  → related to band width / energy spread")
        if third_mean is not None:
            print(f"Third Moment   (⟨ε³⟩): {third_mean:.2f} ± {third_std:.2f}  → indicates asymmetry / skewness of DOS")
        if fourth_mean is not None:
            print(f"Fourth Moment  (⟨ε⁴⟩): {fourth_mean:.2f} ± {fourth_std:.2f}  → measures kurtosis / 'tailedness' of DOS")

        return np.array(ans_image2dmoment)
    

class Regression:
    def __init__(self,
                 atom_fea,
                 nbr_fea,
                 nbr_fea_idx,
                 name_images=None,
                 phys_model='gcnn',
                 model_inx=None,
                 print_freq=1,
                 batch_size=256,
                 num_workers=0,
                 atom_fea_len=64,
                 n_conv=3,
                 h_fea_len=128,
                 n_h=1,
                 optim_algorithm='Adam',
                 lr=0.001,
                 momentum=0.9,
                 weight_decay=0,
                 lr_milestones=[100],
                 resume=None,
                 random_seed=1234,
                 start_epoch=0,
                 tabulated_site_index=None,
                 padding_filter=None,
                 tabulated_hopping_distance=None,
                 tabulated_hopping=None,
                 tabulated_power_ss=None,
                 tabulated_power_ds=None,
                 tabulated_power_dd=None,
                 tabulated_power_gamma_ds=None,
                 tabulated_power_gamma_dd=None,
                 **kwargs
                 ):
        
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize Physical Model
        Moment.__init__(self, phys_model, **kwargs)
        
        target = np.zeros(len(atom_fea))
        
        if name_images is None:
            name_images = np.arange(len(atom_fea))

        dataset = [((torch.Tensor(atom_fea[i].to_dense().numpy()),
                     torch.Tensor(nbr_fea[i].to_dense().numpy()),
                     torch.LongTensor(nbr_fea_idx[i].to_dense().numpy()),
                     torch.LongTensor(padding_filter[i].to_dense().numpy())),
                    torch.DoubleTensor([target[i]]),
                    name_images[i],
                    tabulated_site_index[i])
                   for i in range(len(atom_fea))]
        
        cuda = torch.cuda.is_available()
        
        collate_fn = self.collate_pool
        
        train_loader, val_loader, test_loader =\
            self.get_train_val_test_loader(dataset=dataset,
                                           collate_fn=collate_fn,
                                           batch_size=batch_size,
                                           idx_validation=None,
                                           idx_test=None,
                                           num_workers=num_workers,
                                           pin_memory=cuda,
                                           random_seed=random_seed)
        
        # build model
        structures, _, _, _ = dataset[0]
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
        
        # Initialize the class
        if phys_model == 'moment':
            if cuda:
                tabulated_site_index = torch.from_numpy(tabulated_site_index).cuda()
            
            else:
                tabulated_site_index = torch.from_numpy(tabulated_site_index)
            
            self.tabulated_site_index = tabulated_site_index
            self.tabulated_hopping_distance = tabulated_hopping_distance
            self.tabulated_hopping = tabulated_hopping
            self.tabulated_power_ss = tabulated_power_ss
            self.tabulated_power_ds = tabulated_power_ds
            self.tabulated_power_dd = tabulated_power_dd
            self.tabulated_power_gamma_ds = tabulated_power_gamma_ds
            self.tabulated_power_gamma_dd = tabulated_power_gamma_dd
        
        self.lr = lr
        self.cuda = cuda
        self.phys_model = phys_model
        self.print_freq = print_freq
        self.start_epoch = start_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.best_counter = 0
        self.model_inx = model_inx
    
    def check_loss(self, **kwargs):
        # test best model
        if self.cuda:
            best_checkpoint = torch.load('./data/pretrained/band_moments/model_' + str(self.model_inx) + '.pth.tar')
        else:
            best_checkpoint = torch.load('./data/pretrained/band_moments/model_' + str(self.model_inx) + '.pth.tar', map_location=torch.device('cpu'))
        
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        parm = self.eval_test_model(**kwargs)
        return parm
    
    def eval_test_model(self, **kwargs):
        # switch to evaluate mode
        self.model.eval()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.test_loader):
            with torch.no_grad():
                if self.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 input[3].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True)
                                  for crys_idx in input[4]],
                                 [atom_inx.cuda(non_blocking=True)
                                  for atom_inx in input[5]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3],
                                 input[4],
                                 input[5])
            
            # compute output
            cnn_output, out = self.model(*input_var, batch_cif_ids, self.tabulated_hopping_distance, self.tabulated_hopping, self.tabulated_power_ss, self.tabulated_power_ds, self.tabulated_power_dd, self.tabulated_power_gamma_ds, self.tabulated_power_gamma_dd)
            
            if self.phys_model =='moment':
                output, parm, zeta, crys_fea = Moment.moment(
                    self,
                    cnn_output,
                    out,
                    **dict(**kwargs,
                           batch_cif_ids=batch_cif_ids,
                           crystal_atom_idx=input_var[4]))
        
        return parm
    
    def get_train_val_test_loader(self,
                                  dataset,
                                  idx_validation=None,
                                  idx_test=None,
                                  collate_fn=default_collate,
                                  batch_size=256,
                                  num_workers=0,
                                  pin_memory=False,
                                  random_seed=None):
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
        
        indices = np.arange(len(dataset))
        
        kfold_val = deepcopy(indices)
        kfold_test = deepcopy(indices)
        kfold_train = deepcopy(indices)
        
        val_sampler = SubsetRandomSampler(deepcopy(kfold_val))
        test_sampler = SubsetRandomSampler(deepcopy(kfold_test))
        train_sampler = SubsetRandomSampler(deepcopy(kfold_train))
        
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  pin_memory=pin_memory)
        
        val_loader = DataLoader(dataset, batch_size=batch_size,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                collate_fn=collate_fn,
                                pin_memory=pin_memory)
        
        test_loader = DataLoader(dataset, batch_size=batch_size,
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
        batch_padding_filter = []
        crystal_atom_idx = []
        batch_target = []
        batch_cif_ids = []
        batch_site_ids = []
        base_idx = 0
        
        for i, ((atom_fea, nbr_fea, nbr_fea_idx, padding_filter),
                target, cif_id, site_id)\
                in enumerate(dataset_list):
            n_i = atom_fea.shape[0]  # number of atoms for this crystal
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
            batch_padding_filter.append(padding_filter)
            new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
            crystal_atom_idx.append(new_idx)
            batch_target.append(target)
            batch_cif_ids.append(cif_id)
            batch_site_ids.append(site_id)
            base_idx += n_i
        return (torch.cat(batch_atom_fea, dim=0),
                torch.cat(batch_nbr_fea, dim=0),
                torch.cat(batch_nbr_fea_idx, dim=0),
                torch.cat(batch_padding_filter, dim=0),
                crystal_atom_idx,
                torch.LongTensor(batch_site_ids)),\
            torch.stack(batch_target, dim=0),\
            batch_cif_ids

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

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, padding_filter):
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
        padding_filter_flatten = padding_filter.view(-1)
        total_gated_fea = total_gated_fea.view(-1, self.atom_fea_len*2)
        total_gated_fea_bn1 = self.bn1(total_gated_fea[torch.where(padding_filter_flatten==1)[0]])
        total_gated_fea[torch.where(padding_filter_flatten==1)[0]] = total_gated_fea_bn1
        total_gated_fea = total_gated_fea.view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core * padding_filter[:,:,None], dim=1)
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
        self.atom_fea_len = atom_fea_len
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len + 41, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len, model_num_input)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, padding_filter, crystal_atom_idx, atom_inx, batch_cif_ids, tabulated_hopping_distance, tabulated_hopping, tabulated_power_ss, tabulated_power_ds, tabulated_power_dd, tabulated_power_gamma_ds, tabulated_power_gamma_dd):
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
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx, padding_filter)
        
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_fea[nbr_fea_idx, :]
        
        avg_fea = (atom_nbr_fea[:,:,None,:] + atom_nbr_fea[:,None,:,:]) / 2.0
        
        avg_fea = [avg_fea[idx_map][idx]
                   for idx_map, idx in zip(crystal_atom_idx, atom_inx)]
        
        avg_fea = torch.stack(avg_fea, dim=0)
        
        cuda = torch.cuda.is_available()
        
        if cuda:
            hopping_distance = torch.stack([tabulated_hopping_distance[cif_ids].to_dense() for cif_ids in batch_cif_ids]).cuda()
        else:
            hopping_distance = torch.stack([tabulated_hopping_distance[cif_ids].to_dense() for cif_ids in batch_cif_ids])
        
        total_nbr_fea = torch.cat([avg_fea, hopping_distance], dim=3).float().reshape(-1, 147)
        
        total_nbr_fea = self.conv_to_fc(self.conv_to_fc_softplus(total_nbr_fea))
        total_nbr_fea = self.conv_to_fc_softplus(total_nbr_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                total_nbr_fea = softplus(fc(total_nbr_fea))
        
        out = self.fc_out(total_nbr_fea).reshape(-1,132,132,3)
        
        if cuda:
            hopping = torch.stack([tabulated_hopping[cif_ids].to_dense() for cif_ids in batch_cif_ids]).cuda()
            power_ss = torch.stack([tabulated_power_ss[cif_ids].to_dense() for cif_ids in batch_cif_ids]).cuda()
            power_ds = torch.stack([tabulated_power_ds[cif_ids].to_dense() for cif_ids in batch_cif_ids]).cuda()
            power_dd = torch.stack([tabulated_power_dd[cif_ids].to_dense() for cif_ids in batch_cif_ids]).cuda()
            power_gamma_ds = torch.stack([tabulated_power_gamma_ds[cif_ids].to_dense() for cif_ids in batch_cif_ids]).cuda()
            power_gamma_dd = torch.stack([tabulated_power_gamma_dd[cif_ids].to_dense() for cif_ids in batch_cif_ids]).cuda()
        else:
            hopping = torch.stack([tabulated_hopping[cif_ids].to_dense() for cif_ids in batch_cif_ids])
            power_ss = torch.stack([tabulated_power_ss[cif_ids].to_dense() for cif_ids in batch_cif_ids])
            power_ds = torch.stack([tabulated_power_ds[cif_ids].to_dense() for cif_ids in batch_cif_ids])
            power_dd = torch.stack([tabulated_power_dd[cif_ids].to_dense() for cif_ids in batch_cif_ids])
            power_gamma_ds = torch.stack([tabulated_power_gamma_ds[cif_ids].to_dense() for cif_ids in batch_cif_ids])
            power_gamma_dd = torch.stack([tabulated_power_gamma_dd[cif_ids].to_dense() for cif_ids in batch_cif_ids])
        
        out_power_ss = torch.pow(torch.nn.functional.softplus(out[:,:,:,2]), 2.0/3.5)[:,:,None,:,None] * power_ss
        out_power_ds = torch.nn.functional.softplus(out[:,:,:,2])[:,:,None,:,None] * power_ds
        out_power_dd = torch.pow(torch.nn.functional.softplus(out[:,:,:,2]), 5.0/3.5)[:,:,None,:,None] * power_dd
        out_power_gamma_ds = out[:,:,:,0][:,:,None,:,None] * power_gamma_ds
        out_power_gamma_dd = out[:,:,:,1][:,:,None,:,None] * power_gamma_dd
        
        out = hopping * (out_power_ss + out_power_ds + out_power_dd + out_power_gamma_ds + out_power_gamma_dd)
        
        return out, self.fc_out(total_nbr_fea).reshape(-1,132,132,3)

    def pooling(self, atom_fea, crystal_atom_idx, atom_inx):
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
        
        summed_fea = [atom_fea[idx_map][idx]
                      for idx_map, idx in zip(crystal_atom_idx, atom_inx)]
        
        return torch.stack(summed_fea, dim=0)


class Moment:

    def __init__(self, model_name, **kwargs):
        # Initialize the class
        if model_name == 'moment':
            self.model_num_input = 3
    
    def moment(self, bond_fea, out, **kwargs):
        
        hh2 = [torch.tensordot(h1, h1, ([2, 3], [0, 1])) for h1 in bond_fea]
        hh3 = [torch.tensordot(h2, h1, ([2, 3], [0, 1])) for h2, h1 in zip(hh2, bond_fea)]
        hh4 = [torch.tensordot(h3, h1, ([2, 3], [0, 1])) for h3, h1 in zip(hh3, bond_fea)]
        
        tinnet_m2 = torch.stack([torch.sum(torch.diag(h2[0, 1:, 0, 1:])) for h2 in hh2])
        tinnet_m3 = torch.stack([torch.sum(torch.diag(h3[0, 1:, 0, 1:])) for h3 in hh3])
        tinnet_m4 = torch.stack([torch.sum(torch.diag(h4[0, 1:, 0, 1:])) for h4 in hh4])
        
        idx = kwargs['batch_cif_ids']
        
        cuda = torch.cuda.is_available()
        
        if cuda:
            idx = torch.from_numpy(np.array(idx, dtype=np.float32)).cuda()
        else:
            idx = torch.from_numpy(np.array(idx, dtype=np.float32))
        
        parm = torch.stack((idx,
                            tinnet_m2,
                            tinnet_m3,
                            tinnet_m4)).T
        
        parm = torch.stack((tinnet_m2,
                            tinnet_m3,
                            tinnet_m4)).T
        
        tinnet_m2 = tinnet_m2.view(len(tinnet_m2),-1)
        return tinnet_m2, parm, out, parm


class Features:
    '''
    Parameters
    ----------
    
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    
    Returns
    -------
    
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    '''

    def __init__(self,
                 max_num_nbr=12,
                 radius=8,
                 dmin=0,
                 step=0.2,
                 dict_atom_fea=None):
        
        # Initialize the class
        self.max_num_nbr = max_num_nbr
        self.step = step
        
        # Load superstructure of atom features
        if dict_atom_fea is None:
            self.dict_atom_fea = self.dict_atom_fea_default()
        else:
            self.dict_atom_fea = dict_atom_fea
        
        # Assert
        assert dmin < radius
        assert radius - dmin > self.step

        # Bond feature filter
        self.filter = np.arange(dmin, radius + self.step, self.step)
        
    def feas(self, image):
        
        # Returns pymatgen structure from ASE Atoms.
        try:
            image = AseAtomsAdaptor.get_structure(image)
        except:
            image = image
        
        # atom feature vector
        atom_fea = np.array([self.dict_atom_fea[i] 
                             for i in image.atomic_numbers])

        # VoronoiConnectivity for bond feature vector
        VC = VoronoiConnectivity(image)
        
        all_nbrs = []
        
        # add connectivity for neighbor atoms distances
        conn = VC.connectivity_array
        
        for ii in range(0, conn.shape[0]):
            curnbr = []
            for jj in range(0, conn.shape[1]):
                for kk in range(0, conn.shape[2]):
                    if conn[ii][jj][kk] != 0:
                        curnbr.append([ii, conn[ii][jj][kk]
                                       / np.max(conn[ii]), jj])
                    else:
                        curnbr.append([ii, 0.0, jj])
            all_nbrs.append(np.array(curnbr))
            
        all_nbrs = [sorted(nbrs, key=lambda x: x[1], reverse=True) 
                    for nbrs in all_nbrs]
        
        nbr_fea_idx = np.array([list(map(lambda x: x[2],
                                         nbr[:self.max_num_nbr]))
                                         for nbr in all_nbrs], dtype=np.int64)
        
        nbr_fea = np.array([list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                            for nbr in all_nbrs])

        nbr_fea = np.exp(-(nbr_fea[..., np.newaxis] - self.filter)**2
                         / self.step**2)
        
        return atom_fea, nbr_fea, nbr_fea_idx

    def dict_atom_fea_default(self):
        # Default superstructure of atom features
        atom_fea_dict = {
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0],
            3: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0],
            4: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            11: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            12: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            14: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            17: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            19: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            20: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            21: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            22: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            23: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            24: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            25: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            26: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            27: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            28: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            29: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            30: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            31: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            32: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            33: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            34: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            36: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            37: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            38: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            39: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            40: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            41: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            42: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            43: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            44: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            45: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            46: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            47: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            48: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            49: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            50: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            51: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            52: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            53: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            54: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            55: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            56: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            57: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            58: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            59: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            60: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            61: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            62: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            63: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            64: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            65: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            66: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            67: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            68: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            69: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            70: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            71: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            72: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            73: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            74: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            75: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            76: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            77: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            78: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            79: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            80: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            81: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            82: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            83: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            84: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            85: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            86: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            87: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            88: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            89: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            90: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            91: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            92: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            93: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            94: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            95: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            96: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            97: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            98: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            99: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            100: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        return atom_fea_dict
        
    def material_dict(self):
        # Element name
        # d-band center, eV
        # d-band full width, eV
        # rd, A
        
        reference_data = {'Ag':{'d_cen':-4.1421, 'full_width': 4.5040, 'rd': 0.6606606606606606 },
                          'Au':{'d_cen':-3.6577, 'full_width': 5.8312, 'rd': 0.7607607607607607 },
                          #'Cd':{'d_cen':-8.6701, 'full_width': 3.4283 },
                          'Co':{'d_cen':-1.5905, 'full_width': 6.6969, 'rd': 0.5605605605605606 },
                          'Cr':{'d_cen':-0.0876, 'full_width': 7.2096, 'rd': 0.6306306306306306 },
                          'Cu':{'d_cen':-2.6521, 'full_width': 4.2446, 'rd': 0.4904904904904905 },
                          'Fe':{'d_cen':-0.9278, 'full_width': 7.1120, 'rd': 0.6206206206206206 },
                          #'Hf':{'d_cen': 2.0669, 'full_width':10.4322, },
                          'Ir':{'d_cen':-2.6636, 'full_width': 9.5022, 'rd': 0.8208208208208209 },
                          #'La':{'d_cen': 2.0866, 'full_width': 8.0907, },
                          'Mn':{'d_cen':-0.6036, 'full_width': 7.0846, 'rd': 0.5905905905905906 },
                          'Mo':{'d_cen':-0.0110, 'full_width': 9.0573, 'rd': 0.8608608608608609 },
                          'Nb':{'d_cen': 0.6878, 'full_width': 8.9762, 'rd': 0.9409409409409409 },
                          'Ni':{'d_cen':-1.6686, 'full_width': 5.3541, 'rd': 0.5205205205205206 },
                          'Os':{'d_cen':-1.9693, 'full_width':10.6399, 'rd': 0.8508508508508509 },
                          'Pd':{'d_cen':-2.0870, 'full_width': 5.6793, 'rd': 0.6706706706706707 },
                          'Pt':{'d_cen':-2.6369, 'full_width': 7.6381, 'rd': 0.7907907907907907 },
                          'Re':{'d_cen':-1.0633, 'full_width':10.9953, 'rd': 0.8808808808808809 },
                          'Rh':{'d_cen':-2.0874, 'full_width': 7.3926, 'rd': 0.7207207207207207 },
                          'Ru':{'d_cen':-1.6305, 'full_width': 8.0663, 'rd': 0.7507507507507507 },
                          'Sc':{'d_cen': 1.7032, 'full_width': 5.9997, 'rd': 0.9409409409409409 },
                          'Ta':{'d_cen': 1.1906, 'full_width':10.8913, 'rd': 1.021021021021021  },
                          'Ti':{'d_cen': 1.3243, 'full_width': 6.7421, 'rd': 0.7907907907907907 },
                          'V': {'d_cen': 0.5548, 'full_width': 6.9774, 'rd': 0.6906906906906907 },
                          'W': {'d_cen': 0.1732, 'full_width':10.7373, 'rd': 0.9409409409409409 },
                          'Y': {'d_cen': 2.2707, 'full_width': 7.8737, 'rd': 1.2512512512512513 },
                          #'Zn':{'d_cen':-7.3926, 'full_width': 2.4747, },
                          'Zr':{'d_cen': 1.6485, 'full_width': 8.9050, 'rd': 1.0710710710710711 }}
        
        with open('./data/MaterialDict.pkl', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        data = {k.decode('utf8'): v for k, v in data.items()}
        
        keys = reference_data.keys()
        
        properties = ['d_cen', 'full_width', 'rd']
        
        for key in keys:
            for prop in properties:
                data[key][prop] = reference_data[key][prop]
        
        return data

