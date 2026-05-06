#!/usr/bin/env python
"""TinNet cohesive-energy prediction for transition-metal alloys.

The model embeds a renormalized-atom cohesion theory module in a graph neural
network while preserving the original notebook-facing API.
"""
# This script is adapted from scripts of Jeffrey C. Grossman 
# and Zachary W. Ulissi.

import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap

from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

try:
    from .tinnet_utils import (
        atom_features,
        as_float_tensor,
        as_long_tensor,
        copy_without_adsorbates,
        cohesive_constant_for_symbols,
        flatten_nested,
        data_path,
        load_checkpoint_state,
        normalize_atom_index,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        save_figure,
        signed_color,
        signed_label,
        stack_scalar_parameters,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )
except ImportError:  # Allows running this file directly during debugging.
    from tinnet_utils import (
        atom_features,
        as_float_tensor,
        as_long_tensor,
        copy_without_adsorbates,
        cohesive_constant_for_symbols,
        flatten_nested,
        data_path,
        load_checkpoint_state,
        normalize_atom_index,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        save_figure,
        signed_color,
        signed_label,
        stack_scalar_parameters,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io.ase import AseAtomsAdaptor


class CohesiveEnergy:
    """Predict cohesive energy with the TinNet cohesion-theory module.

    The model returns ensemble predictions and, optionally, interpretable
    contribution terms used by the SHAP explanation routine.
    """
    def __init__(self,
                 image=None,
                 name='Name'):
        self.image = image
        self.name = name
    
    def predict(self,
                image=None,
                return_all_parm=False,
                atom_fea_len=150,
                n_conv=5,
                h_fea_len=128,
                n_h=2,
                batch_size=1024,
                n_ensemble_models=10):
        """Run the cohesive-energy ensemble prediction.

        The architecture hyperparameters are exposed as keyword defaults so the
        notebook API stays simple while advanced users can reproduce or test
        alternative pretrained configurations without editing the source.
        """
        if image is None:
            image = self.image
        if image is None:
            raise ValueError("image must be provided.")

        images = [copy_without_adsorbates(image)]
        pe, vws, atom_indices = self._build_cohesive_constants(images)

        ensemble_results = [
            self._predict_single_model(
                images=images,
                model_inx=model_inx,
                batch_size=batch_size,
                atom_fea_len=atom_fea_len,
                n_conv=n_conv,
                h_fea_len=h_fea_len,
                n_h=n_h,
                pe=pe,
                vws=vws,
                atom_indices=atom_indices,
            )
            for model_inx in range(n_ensemble_models)
        ]

        ech = np.array([result[0] for result in ensemble_results])
        parms = np.vstack([result[1] for result in ensemble_results])

        if not return_all_parm:
            print("Cohesive Energy")
            print("=" * 45)
            print(f"{'Material':<15} {'Cohesive Energy (eV/atom)':>20}")
            print("-" * 45)
            print(f"{self.name:<15} {np.average(ech):>20.2f} ± {np.std(ech):<20.2f}")
            return ech
        return parms

    @staticmethod
    def _build_cohesive_constants(images):
        """Return promotion energies, Wigner-Seitz volumes, and atom indices."""
        pe = []
        vws = []
        atom_indices = []
        offset = 0

        for structure in images:
            symbols = structure.get_chemical_symbols()
            pe_values = cohesive_constant_for_symbols(symbols, "promotion_energy")
            vws_values = cohesive_constant_for_symbols(symbols, "wigner_seitz_volume")

            pe.append(pe_values)
            vws.append(vws_values)
            atom_indices.append(np.arange(offset, offset + len(symbols)))
            offset += len(symbols)

        return np.array(pe, dtype=object), np.array(vws, dtype=object), atom_indices

    @staticmethod
    def _average_tensor(tensor):
        """Return the scalar mean of a tensor as a NumPy/Python value."""
        return np.average(tensor.detach().cpu().numpy())

    def _predict_single_model(self,
                              images,
                              model_inx,
                              batch_size,
                              atom_fea_len,
                              n_conv,
                              h_fea_len,
                              n_h,
                              pe,
                              vws,
                              atom_indices):
        """Evaluate one ensemble member and return scalar energy + SHAP terms."""
        model = Regression(
            images=images,
            data_format='test',
            phys_model='cohesive_energy',
            optim_algorithm='AdamW',
            batch_size=batch_size,
            model_inx=model_inx,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            constant_1=pe,
            constant_2=vws,
            constant_3=atom_indices,
        )

        output, parm = model.predict()
        energy = self._average_tensor(output)
        contribution_terms = np.average(parm.detach().cpu().numpy(), axis=1)
        return energy, np.concatenate((np.atleast_1d(energy), contribution_terms))

    def ech_shap(self,
                 inp_shap):
        return np.atleast_1d(np.sum(inp_shap, axis=-1))


    def gen_shap(self,
                 ref_image,
                 target_image):
        
        ref_image = copy_without_adsorbates(ref_image)
        target_image = copy_without_adsorbates(target_image)

        predicted_parameter_reference = self.predict(ref_image,
                                                     return_all_parm=True)

        predicted_parameter_target = self.predict(target_image,
                                                  return_all_parm=True)
        
        component_shap_values = []
        predicted_ech_reference = []
        predicted_ech_target = []

        for ref_params, target_params in zip(predicted_parameter_reference,
                                             predicted_parameter_target):
            explainer = shap.Explainer(self.ech_shap,
                                       np.atleast_2d(ref_params[1:]))
            shap_values = explainer(np.atleast_2d(target_params[1:])).values

            component_shap_values.append(shap_values.reshape(-1))
            predicted_ech_reference.append(ref_params[0])
            predicted_ech_target.append(target_params[0])

        component_shap_values = np.asarray(component_shap_values)

        return np.vstack((
            np.asarray(predicted_ech_reference).flatten(),
            np.asarray(predicted_ech_target).flatten(),
            component_shap_values[:, 0].flatten(),
            component_shap_values[:, 1].flatten(),
            component_shap_values[:, 2].flatten(),
            component_shap_values[:, 3].flatten(),
        ))
    
    def explain_shap(self,
                     ref_image=None,
                     ref_name='Reference',
                     plot_name='shap',
                     save_fig='png'):
        """Create a compact SHAP waterfall plot for cohesive-energy terms."""
        target_image = self.image
        target_name = self.name
        if target_image is None:
            raise ValueError("target image must be provided through self.image.")

        set_publication_style()
        fig, ax = plt.subplots(figsize=(3.375 * 2.0, 3.375))

        shap_matrix = self.gen_shap(ref_image, target_image)
        shap_mean = np.average(shap_matrix, axis=1)

        base_value = shap_mean[0]
        target_value = shap_mean[1]
        contributions = shap_mean[2:]
        labels = np.array([r'$E_{prom}$', r'$E_{renorm}$', r'$E_{s}$', r'$E_{d}$'])

        order = np.argsort(contributions)
        contributions = contributions[order]
        labels = labels[order]

        starts = base_value + np.r_[0.0, np.cumsum(contributions[:-1])]
        y_positions = np.arange(len(contributions), 0, -1)
        colors = [signed_color(value) for value in contributions]

        for y, start, contribution, color in zip(y_positions, starts, contributions, colors):
            ax.arrow(x=start, y=y, dx=contribution, dy=0, color=color,
                     width=1.0 / 3.0, head_width=1.0 / 3.0,
                     head_length=0.15 * abs(contribution),
                     length_includes_head=True)
            ax.annotate(signed_label(contribution),
                        xy=(-0.12, y), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=color)

        ax.set_ylim([0.5, len(contributions) + 1.5])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(r'$E_{Cohesive}$ (eV/atom)')
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.tick_params('y', length=0, width=0, which='major')

        ax.plot([base_value, base_value], [len(contributions) + 2, 0.5], '--', color='gray', linewidth=1)
        for start, y in zip(starts[1:], y_positions[:-1]):
            ax.plot([start, start], [y + 1.0 / 3.0, y - 1 - 1.0 / 3.0], '--', color='gray', linewidth=1)

        ax.plot([target_value, target_value], [len(contributions) + 2, 0.5], '--', color='orange', linewidth=1)
        ax.annotate(f"{ref_name}\n{base_value:.2f}",
                    xy=(base_value, 1.15), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='gray')
        ax.annotate(f"{target_name}\n{target_value:.2f}",
                    xy=(target_value, 1.05), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='orange')

        fig.tight_layout()
        save_figure(fig, plot_name, save_fig)
        return fig, ax

'''
collection of tight-binding models.

newns_anderson:
'''


class TightBinding:

    def __init__(self, model_name, main_target, **kwargs):
        """Initialize the cohesion-theory module.

        ``main_target`` is kept for compatibility with the original training
        code.  During inference the auxiliary targets are zero-filled because
        only the checkpointed model outputs are used.
        """
        if model_name != 'cohesive_energy':
            raise ValueError(f"Unsupported tight-binding model: {model_name}")

        self.model_num_input = 6
        self.device = torch_device()
        self.cuda = self.device.type == 'cuda'
        self.root_lamb = 0.1

        main_target = np.asarray(main_target, dtype=np.float32).reshape(-1, 1)
        auxiliary_targets = np.zeros((len(main_target), 5), dtype=np.float32)
        self.target = np.hstack((main_target, auxiliary_targets))

        self.pe = as_float_tensor(flatten_nested(kwargs['constant_1']), self.device)
        self.vws = as_float_tensor(flatten_nested(kwargs['constant_2']), self.device)
        self.index = as_float_tensor(flatten_nested(kwargs['constant_3']), self.device)

    def cohesive_energy(self, namodel_in, dos_source, target, **kwargs):
        """Compute cohesive-energy contributions from neural state variables."""
        del dos_source, target  # Inference-only arguments kept for API compatibility.

        e_ren = namodel_in[:, 0]
        alpha, beta, ns, nd, width = [
            torch.nn.functional.softplus(namodel_in[:, idx])
            for idx in range(1, 6)
        ]

        atom_index = kwargs['constant_3'].to(device=self.device, dtype=torch.long)
        structure_atom_indices = kwargs['crys_idx'][0].to(device=self.device, dtype=torch.long)

        pe = self.pe[atom_index]
        vws = self.vws[atom_index]

        conduction_term = 2.1880420859580444e-19 * (1.0 / vws) ** (2.0 / 3.0)
        conduction_energy = conduction_term * alpha * ns ** (2.0 / 3.0)
        d_band_energy = beta * width / 20.0 * nd * (nd - 10.0)

        site_energy = pe + e_ren + conduction_energy + d_band_energy
        output = site_energy[structure_atom_indices].flatten()

        # Rows are the physically interpretable contribution terms used by SHAP.
        parm = torch.stack((pe, e_ren, conduction_energy, d_band_energy))
        return output, parm


class Regression:
    def __init__(self,
                 images,
                 data_format,
                 phys_model='gcnn',
                 batch_size=256,
                 model_inx=None,
                 num_workers=0,
                 random_seed=1234,
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1,
                 # hyperparameters
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
        main_target = [0, 0]
        TightBinding.__init__(self, phys_model, main_target, **kwargs)
        
        # initial settings
        device = torch_device()
        cuda = device.type == 'cuda'
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
        dataset = [((torch.as_tensor(atom_fea[i], dtype=torch.float32),
                     torch.as_tensor(nbr_fea[i], dtype=torch.float32),
                     torch.LongTensor(nbr_fea_idx[i])),
                    torch.as_tensor(np.array(self.target[i]), dtype=torch.float32),
                    idx_images[i],
                    kwargs['constant_1'][i],
                    kwargs['constant_2'][i],
                    kwargs['constant_3'][i])
                   for i in range(len(atom_fea))]
        
        train_loader, val_loader, test_loader =\
            self.get_train_val_test_loader(dataset=dataset,
                                           collate_fn=collate_fn,
                                           batch_size=batch_size,
                                           idx_val_fold=model_inx,
                                           idx_test_fold=model_inx,
                                           train_ratio=train_ratio,
                                           val_ratio=val_ratio,
                                           test_ratio=test_ratio,
                                           num_workers=num_workers,
                                           pin_memory=cuda,
                                           random_seed=random_seed,
                                           data_format=data_format)
        
        # build model
        structures, _, _, _, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len,
                                    nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    h_fea_len=h_fea_len,
                                    n_h=n_h,
                                    model_num_input=self.model_num_input,
                                    idx_val_fold=model_inx,   # Add these
                                    idx_test_fold=model_inx) # Add these
        
        model.to(device)
        
        self.device = device
        self.cuda = cuda
        self.phys_model = phys_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.images = images
        self.model_inx = model_inx

    def predict(self, **kwargs):
        # test best model
        load_checkpoint_state(
            self.model,
            pretrained_path('cohesive_energy', f'model_{self.model_inx}.pth.tar'),
            torch_device(),
        )
        
        output, parm \
            = self.eval_model(catagory='test',
                              data_loader=self.test_loader,
                              save_outputs=True,
                              **kwargs)
        
        return output, parm
    
    def eval_model(self, catagory, data_loader, save_outputs=False, **kwargs):
        """Evaluate the loaded checkpoint on a DataLoader."""
        del save_outputs
        self.model.eval()

        output = parm = None
        with torch.no_grad():
            for batch in data_loader:
                (inputs, target, batch_cif_ids, constant_1, constant_2, constant_3) = batch
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = self._move_inputs_to_device(inputs)
                target_var = target.to(self.device, non_blocking=True)

                cnn_output = self.model(
                    atom_fea,
                    nbr_fea,
                    nbr_fea_idx,
                    crystal_atom_idx,
                    catagory,
                )

                if self.phys_model != 'cohesive_energy':
                    raise ValueError(f"Unsupported physical model: {self.phys_model}")

                output, parm = TightBinding.cohesive_energy(
                    self,
                    cnn_output,
                    dos_source='model',
                    target=target_var,
                    **dict(
                        kwargs,
                        crys_idx=crystal_atom_idx,
                        constant_1=constant_1.to(self.device, non_blocking=True),
                        constant_2=constant_2.to(self.device, non_blocking=True),
                        constant_3=constant_3.to(self.device, non_blocking=True),
                        catagory=catagory,
                        idx_val_fold=self.model_inx,
                        idx_test_fold=self.model_inx,
                    ),
                )

        if output is None or parm is None:
            raise RuntimeError("DataLoader produced no batches.")
        return output, parm

    def _move_inputs_to_device(self, inputs):
        """Move graph tensors and crystal index maps to the model device."""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        return (
            atom_fea.to(self.device, non_blocking=True),
            nbr_fea.to(self.device, non_blocking=True),
            nbr_fea_idx.to(self.device, non_blocking=True),
            [idx.to(self.device, non_blocking=True) for idx in crystal_atom_idx],
        )

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
        
        indices = np.arange(len(dataset))
        n = len(indices)
        
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
        constants = {"constant_1": [], "constant_2": [], "constant_3": []}
        base_idx = 0

        for (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, constant_1, constant_2, constant_3 in dataset_list:
            n_atoms = atom_fea.shape[0]
            atom_offset = torch.arange(n_atoms, dtype=torch.long) + base_idx

            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
            crystal_atom_idx.append(atom_offset)
            batch_target.append(target)
            batch_cif_ids.append(cif_id)
            constants["constant_1"].append(constant_1)
            constants["constant_2"].append(constant_2)
            constants["constant_3"].append(constant_3)
            base_idx += n_atoms

        batch_constant_1 = flatten_nested(constants["constant_1"])
        batch_constant_2 = flatten_nested(constants["constant_2"])
        batch_constant_3 = flatten_nested(constants["constant_3"])

        return (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
        ), torch.stack(batch_target, dim=0), batch_cif_ids, batch_constant_1, batch_constant_2, batch_constant_3


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

        atom_in_fea: (torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: (torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: torch.Tensor shape (N, atom_fea_len)
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
                 model_num_input=1, idx_val_fold=None, idx_test_fold=None):
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
        self.idx_val_fold = idx_val_fold   # Store them
        self.idx_test_fold = idx_test_fold # Store them
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
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, catagory):
        '''
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: (torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: (torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: torch.Tensor shape (N, )
          Atom hidden features after convolution

        '''
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        atom_fea = self.conv_to_fc(self.conv_to_fc_softplus(atom_fea))
        atom_fea = self.conv_to_fc_softplus(atom_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                atom_fea = softplus(fc(atom_fea))
        
        out = self.fc_out(atom_fea)
        
        return out


class Voronoi:
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
    dict_atom_fea: dict
        A dictionary that stores the initialization vector for each element.
    
    Returns
    -------
    
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    '''

    def __init__(self,
                 max_num_nbr=12,
                 radius=8.0,
                 dmin=0.0,
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

