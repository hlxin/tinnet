'''
collection of chemisorption models.

newns_anderson:
'''

import torch
import numpy as np


class Chemisorption:

    def __init__(self, model_name, **kwargs):
        # Initialize the class
        if model_name == 'gcnn':
            self.model_num_input = 1
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
        
        idx = torch.from_numpy(np.array(idx, dtype=np.float32)).cuda()
        
        parm = torch.stack((idx,
                            tinnet_m2,
                            tinnet_m3,
                            tinnet_m4)).T
        
        parm = torch.stack((tinnet_m2,
                            tinnet_m3,
                            tinnet_m4)).T
        
        tinnet_m2 = tinnet_m2.view(len(tinnet_m2),-1)
        return tinnet_m2, parm, out, parm
    
    def gcnn(self, gcnnmodel_in, **kwargs):
        # Do nothing
        return gcnnmodel_in, gcnnmodel_in
