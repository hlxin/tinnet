#!/usr/bin/env python
"""TinNet low-order d-band moment prediction.

The model combines graph neural networks with tight-binding moment theory for
interpretable second-, third-, and fourth-moment predictions.
"""
# This script is adapted from Xie's and Ulissi's scripts.

import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

try:
    from .gnn import ConvLayer
    from .physics import MomentModel
    from .tinnet_utils import (
        atom_features,
        material_properties,
        copy_without_adsorbates,
        load_checkpoint_state,
        pretrained_path,
        torch_device,
        validate_atom_index,
    )
except ImportError:  # Allows running this file directly during debugging.
    from gnn import ConvLayer
    from physics import MomentModel
    from tinnet_utils import (
        atom_features,
        material_properties,
        copy_without_adsorbates,
        load_checkpoint_state,
        pretrained_path,
        torch_device,
        validate_atom_index,
    )


class BandMoments:
    """Predict low-order d-band moments for a selected active site.

    The implementation constructs graph and tight-binding hopping features for
    the selected atom and evaluates the pretrained ensemble.
    """

    N_ENSEMBLE_MODELS = 10
    MOMENT_NAMES = (
        ("Second Moment  (⟨ε²⟩)", "related to band width / energy spread"),
        ("Third Moment   (⟨ε³⟩)", "indicates asymmetry / skewness of DOS"),
        ("Fourth Moment  (⟨ε⁴⟩)", "measures kurtosis / 'tailedness' of DOS"),
    )
    DEFAULT_MODEL_KWARGS = dict(
        lr=0.0044485033567158005,
        atom_fea_len=106,
        n_conv=9,
        h_fea_len=60,
        n_h=2,
        batch_size=16,
    )
    def __init__(self,
                 image=None,
                 atom_inx=None,
                 name='Name'):
        data = material_properties()
        atom_fea_dict = atom_features()
        
        self.data = data
        self.atom_fea_dict = atom_fea_dict
        self.image = image
        self.atom_inx = atom_inx
        self.name = name
    
    def predict(self,
                image=None,
                return_all_parm=False):
        """Predict the second-, third-, and fourth-order d-band moments."""
        input_image = self.image if image is None else image
        atom_inx = self.atom_inx
        system_name = self.name

        if input_image is None:
            raise ValueError("image must be provided.")
        if atom_inx is None:
            raise ValueError("atom_inx must be provided.")

        input_image = copy_without_adsorbates(input_image)
        validate_atom_index(input_image, atom_inx, label="atom_inx")

        images = [input_image.copy(), input_image.copy()]
        atom_inx = np.array([atom_inx, atom_inx])
        
        model_kwargs = dict(self.DEFAULT_MODEL_KWARGS)

        atom_fea_dict = self.atom_fea_dict
        data = self.data
        
    
        atom_fea = [] # atomic features of each site
        nbr_fea_idx = [] # index of neighboring atoms
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
            
            radius = 8.0
            dmin = 0.0
            step = 0.2
            
            assert dmin < radius
            assert radius - dmin > step
            
            bond_feature_filter = np.arange(dmin, radius + step, step)
            
            
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
                
                bond_fea = np.exp(-(nbr_dis[..., np.newaxis]
                                    - bond_feature_filter)**2
                                  / step**2)
                
                nbr_fea_tmp += [np.pad(bond_fea,
                                       [(0, pad_size-len(bond_fea)), (0, 0)],
                                       mode='constant',
                                       constant_values=0) * bond_filter[:,None]]
            
            atom_fea += [torch.tensor(np.array(atom_fea_tmp)).to_sparse_coo()] # atomic features of each site
            nbr_fea_idx += [torch.tensor(np.array(nbr_fea_idx_tmp)).to_sparse_coo()] # index of neighboring atoms
            nbr_fea += [torch.tensor(np.array(nbr_fea_tmp)).to_sparse_coo()] # bond features
            padding_filter += [torch.tensor(np.array(padding_filter_tmp)).to_sparse_coo()] # neighoring atoms within 1 rc (3.6 A)
        
        moment_predictions = self._run_ensemble(
            atom_fea=atom_fea,
            nbr_fea=nbr_fea,
            nbr_fea_idx=nbr_fea_idx,
            atom_inx=atom_inx,
            padding_filter=padding_filter,
            tabulated_hopping_distance=hopping_distance,
            tabulated_hopping=hopping,
            tabulated_power_ss=power_ss,
            tabulated_power_ds=power_ds,
            tabulated_power_dd=power_dd,
            tabulated_power_gamma_ds=power_gamma_ds,
            tabulated_power_gamma_dd=power_gamma_dd,
            model_kwargs=model_kwargs,
        )

        self._print_summary(moment_predictions, int(atom_inx[0]), system_name)
        return moment_predictions

    def _run_ensemble(self, *, atom_fea, nbr_fea, nbr_fea_idx, atom_inx,
                      padding_filter, tabulated_hopping_distance, tabulated_hopping,
                      tabulated_power_ss, tabulated_power_ds, tabulated_power_dd,
                      tabulated_power_gamma_ds, tabulated_power_gamma_dd, model_kwargs):
        """Evaluate all pretrained moment ensemble members."""
        predictions = []
        for model_inx in range(self.N_ENSEMBLE_MODELS):
            model = Regression(
                atom_fea,
                nbr_fea,
                nbr_fea_idx,
                phys_model='moment',
                optim_algorithm='AdamW',
                weight_decay=0.0001,
                model_inx=model_inx,
                tabulated_site_index=atom_inx,
                padding_filter=padding_filter,
                tabulated_hopping_distance=tabulated_hopping_distance,
                tabulated_hopping=tabulated_hopping,
                tabulated_power_ss=tabulated_power_ss,
                tabulated_power_ds=tabulated_power_ds,
                tabulated_power_dd=tabulated_power_dd,
                tabulated_power_gamma_ds=tabulated_power_gamma_ds,
                tabulated_power_gamma_dd=tabulated_power_gamma_dd,
                **model_kwargs,
            )
            parm = model.check_loss()
            predictions.append(parm[0].detach().cpu().numpy())
        return np.asarray(predictions)

    @classmethod
    def _print_summary(cls, predictions, atom_inx, system_name):
        """Print ensemble mean/std for the three low-order moments."""
        mean_moments = np.mean(predictions, axis=0)
        std_moments = np.std(predictions, axis=0)

        print(f"--- Moments for atom index {atom_inx} of {system_name} ---")
        for (label, description), mean, std in zip(cls.MOMENT_NAMES, mean_moments, std_moments):
            print(f"{label}: {mean:.2f} ± {std:.2f}  → {description}")


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
        
        # Inference-only path: autograd anomaly detection is intentionally disabled.
        
        # Initialize Physical Model
        Moment.__init__(self, phys_model, **kwargs)
        
        target = np.zeros(len(atom_fea))
        
        if name_images is None:
            name_images = np.arange(len(atom_fea))

        dataset = [((torch.as_tensor(atom_fea[i].to_dense().numpy(), dtype=torch.float32),
                     torch.as_tensor(nbr_fea[i].to_dense().numpy(), dtype=torch.float32),
                     torch.LongTensor(nbr_fea_idx[i].to_dense().numpy()),
                     torch.LongTensor(padding_filter[i].to_dense().numpy())),
                    torch.as_tensor([target[i]], dtype=torch.float64),
                    name_images[i],
                    tabulated_site_index[i])
                   for i in range(len(atom_fea))]
        
        device = torch_device()
        cuda = device.type == "cuda"
        
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
        
        model.to(device)
        
        # Initialize the theory-module tensors used by the moment model.
        if phys_model == 'moment':
            self.tabulated_site_index = torch.as_tensor(
                tabulated_site_index, dtype=torch.long, device=device
            )
            self.tabulated_hopping_distance = tabulated_hopping_distance
            self.tabulated_hopping = tabulated_hopping
            self.tabulated_power_ss = tabulated_power_ss
            self.tabulated_power_ds = tabulated_power_ds
            self.tabulated_power_dd = tabulated_power_dd
            self.tabulated_power_gamma_ds = tabulated_power_gamma_ds
            self.tabulated_power_gamma_dd = tabulated_power_gamma_dd
        
        self.lr = lr
        self.device = device
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
        load_checkpoint_state(
            self.model,
            pretrained_path('band_moments', f'model_{self.model_inx}.pth.tar'),
            torch_device(),
        )
        
        parm = self.eval_test_model(**kwargs)
        return parm
    
    def _move_batch_to_device(self, batch_input):
        """Move a collated moment-model batch to the active torch device."""
        atom_fea, nbr_fea, nbr_fea_idx, padding_filter, crystal_atom_idx, atom_inx = batch_input
        return (
            atom_fea.to(self.device, non_blocking=True),
            nbr_fea.to(self.device, non_blocking=True),
            nbr_fea_idx.to(self.device, non_blocking=True),
            padding_filter.to(self.device, non_blocking=True),
            [idx.to(self.device, non_blocking=True) for idx in crystal_atom_idx],
            [idx.to(self.device, non_blocking=True) for idx in atom_inx],
        )

    def eval_test_model(self, **kwargs):
        """Evaluate the pretrained moment model on the test loader."""
        self.model.eval()
        parm = None

        with torch.no_grad():
            for batch_input, target, batch_cif_ids in self.test_loader:
                input_var = self._move_batch_to_device(batch_input)
                cnn_output, out = self.model(
                    *input_var,
                    batch_cif_ids,
                    self.tabulated_hopping_distance,
                    self.tabulated_hopping,
                    self.tabulated_power_ss,
                    self.tabulated_power_ds,
                    self.tabulated_power_dd,
                    self.tabulated_power_gamma_ds,
                    self.tabulated_power_gamma_dd,
                )

                if self.phys_model != 'moment':
                    raise ValueError(f"Unsupported physical model: {self.phys_model}")

                _, parm, _, _ = Moment.moment(
                    self,
                    cnn_output,
                    out,
                    **dict(
                        **kwargs,
                        batch_cif_ids=batch_cif_ids,
                        crystal_atom_idx=input_var[4],
                    ),
                )

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
        
    @staticmethod
    def _stack_dense_by_ids(tensors, batch_ids, device):
        """Stack sparse COO tensors selected by batch IDs as dense tensors."""
        return torch.stack([
            tensors[int(batch_id)].to_dense().to(device)
            for batch_id in batch_ids
        ])

    @staticmethod
    def _select_site_pair_features(pair_features, crystal_atom_idx, atom_inx):
        """Select pair features centered at the requested site of each crystal."""
        selected = [
            pair_features[idx_map][idx]
            for idx_map, idx in zip(crystal_atom_idx, atom_inx)
        ]
        return torch.stack(selected, dim=0)

    def _pair_mlp(self, pair_features, hopping_distance):
        """Apply the pair-wise fully connected network to hopping pairs."""
        total_pair_features = torch.cat([pair_features, hopping_distance], dim=3)
        total_pair_features = total_pair_features.float().reshape(-1, 147)
        hidden = self.conv_to_fc(self.conv_to_fc_softplus(total_pair_features))
        hidden = self.conv_to_fc_softplus(hidden)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                hidden = softplus(fc(hidden))
        raw_corrections = self.fc_out(hidden).reshape(-1, 132, 132, 3)
        return raw_corrections, hidden

    def _stack_hopping_terms(self, batch_cif_ids, terms, device):
        """Load all tabulated tight-binding tensors for a mini-batch."""
        return {
            name: self._stack_dense_by_ids(tensor_list, batch_cif_ids, device)
            for name, tensor_list in terms.items()
        }

    _apply_tight_binding_corrections = staticmethod(MomentModel.apply_corrections)

    def forward(
        self,
        atom_fea,
        nbr_fea,
        nbr_fea_idx,
        padding_filter,
        crystal_atom_idx,
        atom_inx,
        batch_cif_ids,
        tabulated_hopping_distance,
        tabulated_hopping,
        tabulated_power_ss,
        tabulated_power_ds,
        tabulated_power_dd,
        tabulated_power_gamma_ds,
        tabulated_power_gamma_dd,
    ):
        """Forward pass for the moment TinNet model."""
        device = atom_fea.device

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx, padding_filter)

        atom_nbr_fea = atom_fea[nbr_fea_idx, :]
        pair_features = (atom_nbr_fea[:, :, None, :] + atom_nbr_fea[:, None, :, :]) / 2.0
        pair_features = self._select_site_pair_features(pair_features, crystal_atom_idx, atom_inx)

        hopping_distance = self._stack_dense_by_ids(
            tabulated_hopping_distance, batch_cif_ids, device
        )
        raw_corrections, hidden = self._pair_mlp(pair_features, hopping_distance)

        terms = self._stack_hopping_terms(
            batch_cif_ids,
            {
                'hopping': tabulated_hopping,
                'power_ss': tabulated_power_ss,
                'power_ds': tabulated_power_ds,
                'power_dd': tabulated_power_dd,
                'power_gamma_ds': tabulated_power_gamma_ds,
                'power_gamma_dd': tabulated_power_gamma_dd,
            },
            device,
        )
        corrected_hopping = self._apply_tight_binding_corrections(raw_corrections, terms)

        return corrected_hopping, self.fc_out(hidden).reshape(-1, 132, 132, 3)

    def pooling(self, atom_fea, crystal_atom_idx, atom_inx):
        '''
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: (torch.Tensor) shape (N, atom_fea_len)
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
    """Adapter between the pretrained moment network and ``MomentModel``."""

    def __init__(self, model_name, physics=None, **kwargs):
        if model_name == 'moment':
            self.physics = MomentModel() if physics is None else physics
            self.model_num_input = self.physics.n_latent

    def moment(self, bond_fea, out, **kwargs):
        """Return ``(m2 (B, 1), moments (B, 3), raw corrections, moments)``."""
        parm = self.physics.moments(bond_fea)
        tinnet_m2 = parm[:, 0].view(len(parm), -1)
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
            self.dict_atom_fea = atom_features()
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


