'''
collection of chemisorption models.

newns_anderson:
'''

import torch
import numpy as np


class Moment:

    def __init__(self, model_name, **kwargs):
        # Initialize the class
        if model_name == 'moment':
            self.model_num_input = 1
    
    def moment(self, bond_fea, crys_fea, **kwargs):
        
        tabulated_filling_inf = self.tabulated_filling_inf
        tabulated_d_cen_inf = self.tabulated_d_cen_inf
        tabulated_full_width_inf = self.tabulated_full_width_inf
        tabulated_mulliken = self.tabulated_mulliken
        tabulated_site_index = self.tabulated_site_index
        tabulated_v2dd = self.tabulated_v2dd
        tabulated_v2ds = self.tabulated_v2ds
        
        zeta = bond_fea[tabulated_site_index][:,0]
        zeta = torch.nn.functional.softplus(zeta)
        
        filling_tinnet = torch.sigmoid(crys_fea[:,2])
        
        alpha = crys_fea[:,0] # elect. transf
        beta = torch.nn.functional.softplus(crys_fea[:,1]) # resonance
        
        crys_fea = torch.stack((alpha, beta)).flatten()
        
        m2 = torch.sum(tabulated_v2ds / zeta
                     + tabulated_v2dd / zeta**(10.0/7.0))
        
        full_width_tinnet = (12*m2)**0.5
        
        d_cen_tinnet = (beta
                        * m2**0.5
                        * (tabulated_d_cen_inf / tabulated_full_width_inf
                           - alpha * tabulated_mulliken))
        
        parm = torch.stack([filling_tinnet,
                            d_cen_tinnet,
                            torch.atleast_1d(full_width_tinnet)])
        
        return d_cen_tinnet, parm, zeta, crys_fea
