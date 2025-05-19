'''
collection of chemisorption models.

newns_anderson:
'''

import csv
import torch
import numpy as np


class Chemisorption:

    def __init__(self, model_name, main_target, **kwargs):
        # Initialize the class
        if model_name == 'gcnn':
            self.model_num_input = 1
            self.target = main_target
            
        if model_name == 'cohesive_energy':
            self.model_num_input = 6
            
            cuda = torch.cuda.is_available()
            
            # target(s)
            root_lamb = 0.1
            
            main_target = np.reshape(main_target,(-1,1))
            
            error_alpha = np.zeros((len(main_target),1))
            error_beta = np.zeros((len(main_target),1))
            error_ns = np.zeros((len(main_target),1))
            error_nd = np.zeros((len(main_target),1))
            error_w = np.zeros((len(main_target),1))
            
            error_alpha = np.reshape(error_alpha,(-1,1))
            error_beta = np.reshape(error_beta,(-1,1))
            error_ns = np.reshape(error_ns,(-1,1))
            error_nd = np.reshape(error_nd,(-1,1))
            error_w = np.reshape(error_w,(-1,1))
                        
            self.target = np.hstack((main_target,
                                     error_alpha,
                                     error_beta,
                                     error_ns,
                                     error_nd,
                                     error_w))
            
            # Initialize the class
            
            pe = kwargs['constant_1']
            vws = kwargs['constant_2']
            index = kwargs['constant_3']
            
            pe = np.array([item for sublist in pe for item in sublist])
            vws = np.array([item for sublist in vws for item in sublist])
            index = np.array([item for sublist in index for item in sublist])

            pe = torch.from_numpy(pe).type(torch.FloatTensor)
            vws = torch.from_numpy(vws).type(torch.FloatTensor)
            index = torch.from_numpy(index).type(torch.FloatTensor)
            
            if cuda:
                pe = pe.cuda()
                vws = vws.cuda()
                index = index.cuda()
            
            self.pe = pe
            self.vws = vws
            self.index = index
            
            self.root_lamb = root_lamb
            
        if model_name == 'user_defined':
            self.model_num_input = 1
            self.target = main_target
        
    def cohesive_energy(self, namodel_in, dos_source, target, **kwargs):
        
        model_e_ren = namodel_in[:,0]
        model_alpha = torch.nn.functional.softplus(namodel_in[:,1])
        model_beta = torch.nn.functional.softplus(namodel_in[:,2])
        model_ns = torch.nn.functional.softplus(namodel_in[:,3])
        model_nd = torch.nn.functional.softplus(namodel_in[:,4])
        model_w = torch.nn.functional.softplus(namodel_in[:,5])
        
        idx = kwargs['crys_idx']
        
        atom_index = kwargs['constant_3'].cuda()
        catagory = kwargs['catagory']
        idx_val_fold = kwargs['idx_val_fold']
        idx_test_fold = kwargs['idx_test_fold']
        
        pe = torch.stack([self.pe[int(idx_map)] for idx_map in atom_index], dim=0)
        vws = torch.stack([self.vws[int(idx_map)] for idx_map in atom_index], dim=0)
        
        parm = torch.stack((atom_index.cuda(),
                            model_e_ren,
                            model_alpha,
                            model_beta,
                            model_ns,
                            model_nd,
                            model_w)).T
        
        h_bar = 1.0545718e-34
        m_ele = 9.10938356e-31
        j_to_ev = 6.242e18
        
        par = 2.1880420859580444e-19 * (1/vws)**(2.0/3.0)
        
        model_energy = (pe
                        + model_e_ren
                        + par * model_alpha*(model_ns)**(2.0/3.0)
                        + model_beta*model_w/20*model_nd*(model_nd-10))
        
        idxs = [atom_index[idx_map] for idx_map in idx]
        '''
        with open('file1_' + str(catagory) + '_val_' + str(idx_val_fold) + '_test_' + str(idx_test_fold) + '.csv', 'a+') as f:
            writer = csv.writer(f)
            for cif_id, pred in zip(idxs, model_energy):
                writer.writerow((cif_id.detach().cpu().numpy(), pred.detach().cpu().numpy()))
        
        with open('file2_' + str(catagory) + '_val_' + str(idx_val_fold) + '_test_' + str(idx_test_fold) + '.csv', 'a+') as f:
            writer = csv.writer(f)
            for cif_id, t1, t2, p1, p2, p3, p4, p5, p6 in zip(atom_index, pe, vws, model_e_ren, model_alpha, model_beta, model_ns, model_nd, model_w):
                writer.writerow((cif_id.detach().cpu().numpy(), t1.detach().cpu().numpy(), t2.detach().cpu().numpy(), t3.detach().cpu().numpy(), t4.detach().cpu().numpy(), t5.detach().cpu().numpy(), t6.detach().cpu().numpy(), t7.detach().cpu().numpy(), p1.detach().cpu().numpy(), p2.detach().cpu().numpy(), p3.detach().cpu().numpy(), p4.detach().cpu().numpy(), p5.detach().cpu().numpy(), p6.detach().cpu().numpy()))
        '''
        ans = model_energy.view(len(model_energy),1,-1)
        return torch.flatten(ans[torch.Tensor.int(idxs[0])]), parm
    
    def gcnn(self, gcnnmodel_in, **kwargs):
        # Do nothing
        return gcnnmodel_in, gcnnmodel_in

    def user_defined(self, user_defined_model_in, **kwargs):
        # Do something:
        return user_defined_model_in, user_defined_model_in
    