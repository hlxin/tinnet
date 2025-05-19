import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


from ..phys.phys import Moment


class Prediction:
    def __init__(self,
                 atom_fea,
                 nbr_fea,
                 nbr_fea_idx,
                 phys_model='moment',
                 idx_model=0,
                 atom_fea_len=106,
                 n_conv=9,
                 h_fea_len=60,
                 n_h=2,
                 tabulated_filling_inf=None,
                 tabulated_d_cen_inf=None,
                 tabulated_padding_fillter=None,
                 tabulated_full_width_inf=None,
                 tabulated_mulliken=None,
                 tabulated_site_index=None,
                 tabulated_v2dd=None,
                 tabulated_v2ds=None,
                 **kwargs
                 ):
        
        # Initialize Physical Model
        Moment.__init__(self, phys_model, **kwargs)
        
        cuda = False #torch.cuda.is_available()
        
        # build model
        orig_atom_fea_len = atom_fea.shape[-1]
        nbr_fea_len = nbr_fea.shape[-1]
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
        if cuda:
            tabulated_filling_inf = torch.from_numpy(np.array(tabulated_filling_inf)).cuda()
            tabulated_d_cen_inf = torch.from_numpy(np.array(tabulated_d_cen_inf)).cuda()
            tabulated_full_width_inf = torch.from_numpy(np.array(tabulated_full_width_inf)).cuda()
            tabulated_mulliken = torch.from_numpy(np.array(tabulated_mulliken)).cuda()
            tabulated_site_index = torch.from_numpy(np.array(tabulated_site_index)).cuda()
            tabulated_v2dd = torch.from_numpy(np.array(tabulated_v2dd)).cuda()
            tabulated_v2ds = torch.from_numpy(np.array(tabulated_v2ds)).cuda()
        
        else:
            tabulated_filling_inf = torch.from_numpy(np.array(tabulated_filling_inf))
            tabulated_d_cen_inf = torch.from_numpy(np.array(tabulated_d_cen_inf))
            tabulated_full_width_inf = torch.from_numpy(np.array(tabulated_full_width_inf))
            tabulated_mulliken = torch.from_numpy(np.array(tabulated_mulliken))
            tabulated_site_index = torch.from_numpy(np.array(tabulated_site_index))
            tabulated_v2dd = torch.from_numpy(np.array(tabulated_v2dd))
            tabulated_v2ds = torch.from_numpy(np.array(tabulated_v2ds))
        
        self.tabulated_filling_inf = tabulated_filling_inf
        self.tabulated_d_cen_inf = tabulated_d_cen_inf
        self.tabulated_full_width_inf = tabulated_full_width_inf
        self.tabulated_mulliken = tabulated_mulliken
        self.tabulated_site_index = tabulated_site_index
        self.tabulated_v2dd = tabulated_v2dd
        self.tabulated_v2ds = tabulated_v2ds
        
        self.cuda = cuda
        self.phys_model = phys_model
        self.model = model
        self.idx_model = idx_model
        
        self.atom_fea = torch.from_numpy(atom_fea.astype(np.float32))
        self.nbr_fea = torch.from_numpy(nbr_fea.astype(np.float32))
        self.nbr_fea_idx = torch.from_numpy(nbr_fea_idx)
        self.tabulated_padding_fillter = torch.from_numpy(tabulated_padding_fillter)
        self.crystal_atom_idx = np.arange(atom_fea.shape[0])
    
    def predict_d_cen(self,
                      return_all_parm=False,
                      **kwargs):
        
        if self.cuda:
            best_checkpoint = torch.load('./d_center/tinnet/pretrained/model_best_train_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.pth.tar')
        else:
            best_checkpoint = torch.load('./d_center/tinnet/pretrained/model_best_train_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.pth.tar', map_location=torch.device('cpu'))
        
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        # switch to evaluate mode
        self.model.eval()
        
        with torch.no_grad():
            if self.cuda:
                input_var = (Variable(self.atom_fea.cuda(non_blocking=True)),
                             Variable(self.nbr_fea.cuda(non_blocking=True)),
                             self.nbr_fea_idx.cuda(non_blocking=True),
                             self.tabulated_padding_fillter.cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True)
                              for crys_idx in self.crystal_atom_idx],
                             [site_idx.cuda(non_blocking=True)
                              for site_idx in self.tabulated_site_index])
            else:
                input_var = (Variable(self.atom_fea),
                             Variable(self.nbr_fea),
                             self.nbr_fea_idx,
                             self.tabulated_padding_fillter,
                             self.crystal_atom_idx,
                             self.tabulated_site_index)
        
        # compute output
        cnn_output, cnn_output_crys = self.model(*input_var)
        
        if self.phys_model =='moment':
            output, parm, zeta, crys_fea = Moment.moment(
                self,
                cnn_output,
                cnn_output_crys)
        
        #np.savetxt('parm_test_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.txt', parm.detach().cpu().numpy())
        
        #np.save('zeta_test_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.npy', zeta.detach().cpu().numpy())
        
        #np.save('crys_fea_test_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.npy', crys_fea.detach().cpu().numpy())
        
        if return_all_parm == True:
            return (output.detach().cpu().numpy(),
                    parm.detach().cpu().numpy(),
                    zeta.detach().cpu().numpy()**(1.0/7.0),
                    crys_fea.detach().cpu().numpy())
        else:
            return output.detach().cpu().numpy()
    
    def predict_properties(self,
                      return_all_parm=False,
                      **kwargs):
        
        if self.cuda:
            best_checkpoint = torch.load('./d_center/tinnet/pretrained/model_best_train_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.pth.tar')
        else:
            best_checkpoint = torch.load('./d_center/tinnet/pretrained/model_best_train_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.pth.tar', map_location=torch.device('cpu'))
        
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        # switch to evaluate mode
        self.model.eval()
        
        with torch.no_grad():
            if self.cuda:
                input_var = (Variable(self.atom_fea.cuda(non_blocking=True)),
                             Variable(self.nbr_fea.cuda(non_blocking=True)),
                             self.nbr_fea_idx.cuda(non_blocking=True),
                             self.tabulated_padding_fillter.cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True)
                              for crys_idx in self.crystal_atom_idx],
                             [site_idx.cuda(non_blocking=True)
                              for site_idx in self.tabulated_site_index])
            else:
                input_var = (Variable(self.atom_fea),
                             Variable(self.nbr_fea),
                             self.nbr_fea_idx,
                             self.tabulated_padding_fillter,
                             self.crystal_atom_idx,
                             self.tabulated_site_index)
        
        # compute output
        cnn_output, cnn_output_crys = self.model(*input_var)
        
        if self.phys_model =='moment':
            output, parm, zeta, crys_fea = Moment.moment(
                self,
                cnn_output,
                cnn_output_crys)
        
        #np.savetxt('parm_test_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.txt', parm.detach().cpu().numpy())
        
        #np.save('zeta_test_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.npy', zeta.detach().cpu().numpy())
        
        #np.save('crys_fea_test_idx_val_' + str(self.idx_model) + '_idx_test_' + str(self.idx_model) + '.npy', crys_fea.detach().cpu().numpy())
        
        if return_all_parm == True:
            return (output.detach().cpu().numpy(),
                    parm.detach().cpu().numpy(),
                    zeta.detach().cpu().numpy()**(1.0/7.0),
                    crys_fea.detach().cpu().numpy())
        else:
            return parm.detach().cpu().numpy()

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

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter):
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
        tabulated_padding_fillter_flatten = tabulated_padding_fillter.view(-1)
        total_gated_fea = total_gated_fea.view(-1, self.atom_fea_len*2)
        total_gated_fea_bn1 = self.bn1(total_gated_fea[torch.where(tabulated_padding_fillter_flatten==1)[0]])
        total_gated_fea[torch.where(tabulated_padding_fillter_flatten==1)[0]] = total_gated_fea_bn1
        total_gated_fea = total_gated_fea.view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core * tabulated_padding_fillter[:,:,None], dim=1)
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
        self.conv_to_fc = nn.Linear(atom_fea_len + nbr_fea_len, h_fea_len)
        self.conv_to_fc_crys = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        if n_h > 1:
            self.fcs_crys = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses_crys = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len, model_num_input)
        self.fc_out_crys = nn.Linear(h_fea_len, 3)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter, crystal_atom_idx, site_idx):
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
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter)
        
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_fea[nbr_fea_idx, :]
        
        avg_fea = ((atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len)
                    + atom_nbr_fea) / 2.0)
        
        total_nbr_fea = torch.cat([avg_fea, nbr_fea], dim=2)
        
        total_nbr_fea = self.conv_to_fc_softplus(total_nbr_fea)
        total_nbr_fea = self.conv_to_fc(total_nbr_fea)
        total_nbr_fea = self.conv_to_fc_softplus(total_nbr_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                total_nbr_fea = softplus(fc(total_nbr_fea))
        
        out = self.fc_out(total_nbr_fea) * tabulated_padding_fillter[:,:,None]
        
        crys_fea = torch.atleast_2d(atom_fea[site_idx])
        crys_fea = self.conv_to_fc_crys(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc_crys, softplus_crys in zip(self.fcs_crys,
                                              self.softpluses_crys):
                crys_fea = softplus_crys(fc_crys(crys_fea))
        
        out_crys = self.fc_out_crys(crys_fea)
        
        return out, out_crys
