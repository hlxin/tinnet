#!/usr/bin/env python
"""TinNet d-band filling, center, and rectangular-width prediction.

The implementation preserves the trained model architecture and equations while
improving input validation, checkpoint loading, and plotting utilities.
"""
# This script is adapted from Xie's and Ulissi's scripts.

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn

try:
    from .tinnet_utils import (
        atom_features,
        as_float_tensor,
        material_properties,
        as_long_tensor,
        copy_without_adsorbates,
        data_path,
        load_checkpoint_state,
        normalize_atom_index,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        signed_label,
        signed_color,
        save_figure,
        stack_scalar_parameters,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )
except ImportError:  # Allows running this file directly during debugging.
    from tinnet_utils import (
        atom_features,
        as_float_tensor,
        material_properties,
        as_long_tensor,
        copy_without_adsorbates,
        data_path,
        load_checkpoint_state,
        normalize_atom_index,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        signed_label,
        signed_color,
        save_figure,
        stack_scalar_parameters,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )



class BandCenter:
    """Predict d-band filling, center, and full rectangular width.

    This class wraps an ensemble of pretrained TinNet checkpoints.  The public
    notebook-facing API is kept compatible with the original implementation,
    while the repeated feature-construction and SHAP logic is centralized into
    reusable helper methods.
    """

    N_ENSEMBLE_MODELS = 10
    MAX_NEIGHBORS = 86
    REPEAT_TIMES = 11
    NEIGHBOR_CUTOFF = 5.5

    PROPERTY_INDEX = {
        'filling': 0,
        'center': 1,
        'width': 2,
    }

    SHAP_LABELS = [
        r'$(\alpha, \xi)$',
        r'$d_{ij}$',
        r'$\zeta$',
        r'$(\lambda, r_{dj})$',
        r'$(\beta, \Delta\chi)$',
    ]

    def __init__(self,
                 image=None,
                 atom_inx=None,
                 name='Name'):
        self.descriptor = Features(radius=8,
                                   dmin=0,
                                   step=0.2,
                                   dict_atom_fea=None)
        self.material_dict = material_properties()
        self.image = image
        self.atom_inx = atom_inx
        self.name = name

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------
    def predict(self,
                image=None,
                atom_inx=None,
                return_all_parm=False):
        """Predict the d-band center for the configured atom site."""
        input_image = self.image if image is None else image
        site_index = self.atom_inx if atom_inx is None else atom_inx
        if input_image is None:
            raise ValueError("image must be provided.")
        if site_index is None:
            raise ValueError("atom_inx must be provided.")

        result = self._predict_property(input_image,
                                        site_index,
                                        property_name='center',
                                        return_all_parm=return_all_parm)
        if return_all_parm:
            return result

        mean_center = np.mean(result)
        std_center = np.std(result)
        print(
            f"band center of the atom (index {site_index}) of {self.name}: "
            f"{mean_center:.2f} ± {std_center:.2f} eV"
        )
        return result

    def image2band_filling(self,
                           input_image,
                           atom_inx,
                           return_all_parm=False):
        """Predict d-band filling for a site in an ASE Atoms object."""
        result = self._predict_property(input_image,
                                        atom_inx,
                                        property_name='filling',
                                        return_all_parm=return_all_parm)
        if not return_all_parm:
            print(f"band filling: {np.mean(result):.6f} ± {np.std(result):.6f}")
        return result

    def image2band_center(self,
                          image,
                          atom_inx,
                          return_all_parm=False):
        """Predict d-band center for a site in an ASE Atoms object."""
        return self._predict_property(image,
                                      atom_inx,
                                      property_name='center',
                                      return_all_parm=return_all_parm)

    def image2band_full_rectangular_width(self,
                                          image,
                                          atom_inx,
                                          return_all_parm=False):
        """Predict the full rectangular d-band width for a site."""
        return self._predict_property(image,
                                      atom_inx,
                                      property_name='width',
                                      return_all_parm=return_all_parm)

    def image2dcen(self,
                   image,
                   atom_inx,
                   return_all_parm=False):
        """Predict d-band center through the analytical TinNet theory module.

        ``return_all_parm=True`` preserves the original SHAP-facing return
        value: a list of ensemble model outputs and the tabulated physical
        parameters used as SHAP inputs.
        """
        context = self._prepare_site_context(image, atom_inx)
        predictions = self._run_ensemble(context, method='predict_d_cen',
                                         return_all_parm=return_all_parm)

        if return_all_parm:
            return predictions, context['tabulated_parameters']

        predictions = np.stack(predictions)
        print(f"band center: {np.mean(predictions):.6f} ± {np.std(predictions):.6f}")
        return predictions

    # ------------------------------------------------------------------
    # Shared feature preparation and model execution
    # ------------------------------------------------------------------
    @staticmethod
    def _pad(values, length, constant=0.0):
        """Pad a one-dimensional array to a fixed length."""
        values = np.asarray(values)
        if len(values) > length:
            raise ValueError(
                f"Expected at most {length} neighbors, but found {len(values)}."
            )
        return np.pad(values, [0, length - len(values)],
                      mode='constant', constant_values=constant)

    @staticmethod
    def _mulliken(material_dict, symbol):
        """Return Mulliken electronegativity from tabulated atomic properties."""
        props = material_dict[symbol]
        return (props[b'IonizationPotential'] + props[b'ElectronAffinity']) / 2.0

    def _prepare_site_context(self, image, atom_inx):
        """Build all tabulated inputs required by a BandCenter ensemble model.

        The original script duplicated this block in ``predict``,
        ``image2band_filling``, ``image2band_center``,
        ``image2band_full_rectangular_width``, and ``image2dcen``.  This method
        keeps a single source of truth for neighbor selection, tabulated
        tight-binding features, descriptor construction, and SHAP parameters.
        """
        if image is None:
            raise ValueError("image must be provided.")
        atom_inx = normalize_atom_index(atom_inx)
        clean_image = copy_without_adsorbates(image)
        validate_atom_index(clean_image, atom_inx, label='atom_inx')

        enlarged_image = clean_image.copy()
        enlarged_atom_inx = len(enlarged_image) * 60 + atom_inx
        enlarged_image = enlarged_image.repeat((self.REPEAT_TIMES,
                                                self.REPEAT_TIMES,
                                                1))

        neighbor_distances = enlarged_image.get_distances(
            enlarged_atom_inx, list(range(len(enlarged_image)))
        )
        neighbor_indices = np.where(
            (0.01 <= neighbor_distances) &
            (neighbor_distances <= self.NEIGHBOR_CUTOFF)
        )[0]
        order = np.argsort(neighbor_distances[neighbor_indices])
        neighbor_indices = neighbor_indices[order]
        neighbor_distances = neighbor_distances[neighbor_indices]

        site_symbol = enlarged_image[enlarged_atom_inx].symbol
        neighbor_symbols = np.array([enlarged_image[i].symbol for i in neighbor_indices])

        site_props = self.material_dict[site_symbol]
        tabulated_filling_inf = site_props['bulk_filling']
        tabulated_d_cen_inf = site_props['d_cen']
        tabulated_full_width_inf = site_props['full_width']

        site_radius = site_props['rd']
        neighbor_radii = np.array([self.material_dict[sym]['rd']
                                   for sym in neighbor_symbols])

        vds = site_radius**1.5 / neighbor_distances**3.5
        vdd = site_radius**1.5 * neighbor_radii**1.5 / neighbor_distances**5.0
        tabulated_v2ds = self._pad(9.9856 * vds**2.0 * 7.62**2,
                                   self.MAX_NEIGHBORS)
        tabulated_v2dd = self._pad(415.565 * vdd**2.0 * 7.62**2,
                                   self.MAX_NEIGHBORS)
        tabulated_d_ij = self._pad(neighbor_distances,
                                   self.MAX_NEIGHBORS,
                                   constant=1.0e6)
        tabulated_neighbor_indices = self._pad(neighbor_indices,
                                               self.MAX_NEIGHBORS,
                                               constant=1.0e6)

        (shorten_idx_syms,
         shorten_nbr_syms,
         shorten_atom_fea,
         shorten_nbr_fea,
         shorten_nbr_fea_idx,
         tabulated_d_ij_sorted,
         tabulated_nbr_index_sorted,
         tabulated_v2dd_sorted,
         tabulated_v2ds_sorted,
         tabulated_padding_filter) = self.descriptor.feas(
             clean_image,
             enlarged_image,
             tabulated_neighbor_indices,
             tabulated_d_ij,
             enlarged_atom_inx,
             tabulated_v2ds,
             tabulated_v2dd,
         )

        # Mulliken electronegativity difference for the first neighbor shell.
        all_distances = enlarged_image.get_distances(
            enlarged_atom_inx, list(range(len(enlarged_image)))
        )
        first_shell_cutoff = np.sqrt(1.0) * np.sort(all_distances)[1] + 0.01
        first_shell_indices = np.where(
            (0.01 <= all_distances) & (all_distances <= first_shell_cutoff)
        )[0]
        first_shell_symbols = [enlarged_image[i].symbol for i in first_shell_indices]
        site_mulliken = self._mulliken(self.material_dict, site_symbol)
        neighbor_mulliken = np.array([
            self._mulliken(self.material_dict, sym) for sym in first_shell_symbols
        ])
        tabulated_mulliken = site_mulliken - np.prod(neighbor_mulliken)**(1 / len(neighbor_mulliken))

        padded_neighbor_radii = self._pad(neighbor_radii,
                                          self.MAX_NEIGHBORS,
                                          constant=0.0)
        tabulated_parameters = np.concatenate((
            [site_radius],
            padded_neighbor_radii,
            tabulated_d_ij_sorted,
            [tabulated_d_cen_inf],
            [tabulated_full_width_inf],
            [tabulated_mulliken],
            tabulated_padding_filter[atom_inx],
        ))

        return {
            'atom_fea': shorten_atom_fea,
            'nbr_fea': shorten_nbr_fea,
            'nbr_fea_idx': shorten_nbr_fea_idx,
            'tabulated_filling_inf': tabulated_filling_inf,
            'tabulated_d_cen_inf': tabulated_d_cen_inf,
            'tabulated_full_width_inf': tabulated_full_width_inf,
            'tabulated_mulliken': tabulated_mulliken,
            'tabulated_site_index': np.mod(enlarged_atom_inx, len(clean_image)),
            'tabulated_v2dd': tabulated_v2dd_sorted,
            'tabulated_v2ds': tabulated_v2ds_sorted,
            'tabulated_padding_filter': tabulated_padding_filter,
            'tabulated_parameters': tabulated_parameters,
        }

    def _build_model(self, context, idx_model):
        """Instantiate one pretrained ensemble member from prepared features."""
        return Prediction(
            context['atom_fea'],
            context['nbr_fea'],
            context['nbr_fea_idx'],
            idx_model=idx_model,
            tabulated_filling_inf=context['tabulated_filling_inf'],
            tabulated_d_cen_inf=context['tabulated_d_cen_inf'],
            tabulated_padding_fillter=context['tabulated_padding_filter'],
            tabulated_full_width_inf=context['tabulated_full_width_inf'],
            tabulated_mulliken=context['tabulated_mulliken'],
            tabulated_site_index=context['tabulated_site_index'],
            tabulated_v2dd=context['tabulated_v2dd'],
            tabulated_v2ds=context['tabulated_v2ds'],
        )

    def _run_ensemble(self, context, method, return_all_parm=False):
        """Run all ensemble members with a selected prediction method."""
        outputs = []
        for idx_model in range(self.N_ENSEMBLE_MODELS):
            model = self._build_model(context, idx_model)
            outputs.append(getattr(model, method)(return_all_parm=return_all_parm))
        return outputs

    def _predict_property(self, image, atom_inx, property_name, return_all_parm=False):
        """Shared implementation for filling, center, and width prediction."""
        if property_name not in self.PROPERTY_INDEX:
            raise ValueError(f"Unknown property_name: {property_name}")

        context = self._prepare_site_context(image, atom_inx)
        predictions = self._run_ensemble(context,
                                         method='predict_properties',
                                         return_all_parm=return_all_parm)
        if return_all_parm:
            return predictions, context['tabulated_parameters']

        predictions = np.stack(predictions)[:, :, 0]
        return predictions[:, self.PROPERTY_INDEX[property_name]]

    # ------------------------------------------------------------------
    # SHAP analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _shap_input(tabulated_parameters, model_parameters):
        """Build the 350-dimensional SHAP input vector for one ensemble member."""
        return np.atleast_2d(np.hstack((
            tabulated_parameters,
            model_parameters[2],   # zeta for 86 neighbor entries
            model_parameters[3],   # beta and alpha neural state variables
        )))

    @staticmethod
    def _aggregate_d_center_shap(shap_values):
        """Aggregate raw SHAP values into physical effect categories.

        Input feature layout:
        0       : site d-orbital radius
        1:87    : neighbor d-orbital radii       -> ligand effect
        87:173  : sorted interatomic distances   -> strain effect
        173     : tabulated bulk d-band center
        174     : tabulated bulk full width
        175     : Mulliken electronegativity diff -> charge transfer
        176:262 : padding/filter flags
        262:348 : zeta relaxation coefficients   -> local relaxation
        348     : beta charge-transfer variable   -> charge transfer
        349     : alpha resonance variable        -> resonance
        """
        raw = np.asarray(shap_values)
        return np.array([
            raw[:, 349].sum(),                 # resonance
            raw[:, 87:173].sum(),              # strain: distances
            raw[:, 262:348].sum(),             # relaxation: zeta
            raw[:, 1:87].sum(),                # ligand: neighbor radii
            raw[:, 348].sum() + raw[:, 175].sum(),  # charge transfer
        ])

    def gen_shap(self,
                 ref_image,
                 ref_atom_inx,
                 target_image,
                 target_atom_inx):
        """Return ensemble SHAP decomposition for the d-band center."""
        predicted_ref, tabulated_ref = self.image2dcen(ref_image,
                                                       ref_atom_inx,
                                                       return_all_parm=True)
        predicted_target, tabulated_target = self.image2dcen(target_image,
                                                             target_atom_inx,
                                                             return_all_parm=True)

        rows = []
        for ref_params, target_params in zip(predicted_ref, predicted_target):
            inp_ref = self._shap_input(tabulated_ref, ref_params)
            inp_target = self._shap_input(tabulated_target, target_params)
            explainer = shap.Explainer(self.tinnet_d_center, inp_ref)
            shap_values = explainer(inp_target).values
            rows.append(np.concatenate((
                np.atleast_1d(ref_params[0]).ravel()[:1],
                np.atleast_1d(target_params[0]).ravel()[:1],
                self._aggregate_d_center_shap(shap_values),
            )))

        return np.asarray(rows).T

    def tinnet_d_center(self, inp_shap):
        """Evaluate the analytical d-band-center expression used by SHAP."""
        inp = np.asarray(inp_shap, dtype=float)
        site_radius = inp[:, 0]
        neighbor_radii = inp[:, 1:87]
        distances = inp[:, 87:173]
        bulk_center = inp[:, 173]
        bulk_width = inp[:, 174]
        mulliken_diff = inp[:, 175]
        padding_filter = inp[:, 176:262]
        zeta = inp[:, 262:348]
        beta = inp[:, 348]
        alpha = inp[:, 349]

        vds = site_radius[:, None]**1.5 / distances**3.5
        vdd = site_radius[:, None]**1.5 * neighbor_radii**1.5 / distances**5.0
        v2ds = 9.9856 * vds**2.0 * 7.62**2 * padding_filter
        v2dd = 415.565 * vdd**2.0 * 7.62**2 * padding_filter

        second_moment = np.sum(v2ds / zeta**7.0 + v2dd / zeta**10.0, axis=1)
        d_center = (
            alpha
            * np.sqrt(second_moment)
            * (bulk_center / bulk_width - beta * mulliken_diff)
        )
        return np.atleast_1d(d_center)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    _signed_color = staticmethod(signed_color)
    _format_signed = staticmethod(signed_label)

    def explain_shap(self,
                     ref_image=None,
                     ref_atom_inx=None,
                     ref_name='Reference',
                     plot_name='shap',
                     save_fig='png'):
        """Plot a compact waterfall-style SHAP explanation for d-band center."""
        target_image = self.image
        target_atom_inx = self.atom_inx
        target_name = self.name
        set_publication_style()

        shap_matrix = self.gen_shap(ref_image,
                                    ref_atom_inx,
                                    target_image,
                                    target_atom_inx)
        shap_mean = np.average(shap_matrix, axis=1)
        baseline = shap_mean[0]
        target_prediction = shap_mean[1]
        deltas = shap_mean[2:]

        starts = baseline + np.concatenate(([0.0], np.cumsum(deltas[:-1])))
        y_positions = np.arange(len(deltas), 0, -1)

        fig, ax = plt.subplots()
        fig.set_size_inches(3.375 * 2.0, 3.375)

        for start, delta, y in zip(starts, deltas, y_positions):
            color = self._signed_color(delta)
            ax.arrow(x=start,
                     y=y,
                     dx=delta,
                     dy=0,
                     color=color,
                     width=1.0 / 3.0,
                     head_width=1.0 / 3.0,
                     head_length=0.15 * abs(delta),
                     length_includes_head=True)
            ax.annotate(self._format_signed(delta),
                        xy=(-0.12, y - 0.30),
                        xycoords=('axes fraction', 'data'),
                        ha='center',
                        va='center',
                        color=color)

        ax.set_ylim([0.5, len(deltas) + 0.5])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(self.SHAP_LABELS)

        # Physical-group labels from the original figure, preserved in a compact loop.
        group_annotations = [
            ('Resonance', 5.0, None),
            ('Ligand', 2.0, None),
            ('Charge\ntransfer', 1.0, None),
            ('Strain', 3.5, dict(arrowstyle='-[, widthB=3.0, lengthB=1.0')),
        ]
        for label, y, arrowprops in group_annotations:
            if arrowprops is None:
                ax.annotate(label, xy=(-0.12, y), xycoords=('axes fraction', 'data'),
                            ha='center', va='center')
            else:
                ax.annotate(label,
                            xy=(-0.06, y), xycoords=('axes fraction', 'data'),
                            xytext=(-0.12, y), textcoords=('axes fraction', 'data'),
                            arrowprops=arrowprops,
                            ha='center', va='center')

        ax.set_xlabel(r'$d\rm{-band\ center}$ (eV)')
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.tick_params('y', length=0, width=0, which='major')

        # Dashed connector lines between consecutive waterfall steps.
        ax.plot([baseline, baseline], [len(deltas) + 2, 0.5], '--', color='gray', linewidth=1)
        for x, y in zip(starts[1:], y_positions[:-1]):
            ax.plot([x, x], [y + 1.0 / 3.0, y - 1 - 1.0 / 3.0],
                    '--', color='gray', linewidth=1)
        ax.plot([target_prediction, target_prediction],
                [len(deltas) + 2, 0.5], '--', color='orange', linewidth=1)

        ax.annotate(ref_name + f'\n{baseline:.2f}',
                    xy=(baseline, 1.15),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='center',
                    color='gray')
        ax.annotate(target_name + f'\n{target_prediction:.2f}',
                    xy=(target_prediction, 1.05),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='center',
                    color='orange')

        fig.tight_layout()
        save_figure(fig, plot_name, save_fig)
        return fig, ax

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
        
        device = torch_device()
        cuda = device.type == 'cuda'

        # build model
        orig_atom_fea_len = atom_fea.shape[-1]
        nbr_fea_len = nbr_fea.shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len,
                                    nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    h_fea_len=h_fea_len,
                                    n_h=n_h,
                                    model_num_input=self.model_num_input).to(device)

        def tensor_on_device(value, dtype=torch.float32):
            return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

        # Initialize the class
        tabulated_filling_inf = tensor_on_device(tabulated_filling_inf)
        tabulated_d_cen_inf = tensor_on_device(tabulated_d_cen_inf)
        tabulated_full_width_inf = tensor_on_device(tabulated_full_width_inf)
        tabulated_mulliken = tensor_on_device(tabulated_mulliken)
        tabulated_site_index = tensor_on_device(tabulated_site_index, dtype=torch.long)
        tabulated_v2dd = tensor_on_device(tabulated_v2dd)
        tabulated_v2ds = tensor_on_device(tabulated_v2ds)
        
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
        
        self.device = device
        self.atom_fea = torch.as_tensor(atom_fea.astype(np.float32), device=device)
        self.nbr_fea = torch.as_tensor(nbr_fea.astype(np.float32), device=device)
        self.nbr_fea_idx = torch.as_tensor(nbr_fea_idx, dtype=torch.long, device=device)
        self.tabulated_padding_fillter = torch.as_tensor(tabulated_padding_fillter, device=device)
        self.crystal_atom_idx = torch.as_tensor(np.arange(atom_fea.shape[0]), dtype=torch.long, device=device)
    
    def _input_var(self):
        """Return tensors in the order expected by the GCNN forward pass."""
        return (
            self.atom_fea,
            self.nbr_fea,
            self.nbr_fea_idx,
            self.tabulated_padding_fillter,
            self.crystal_atom_idx,
            self.tabulated_site_index,
        )

    def _predict_moment_outputs(self):
        """Run one pretrained band-center ensemble member and return raw outputs."""
        load_checkpoint_state(
            self.model,
            pretrained_path('band_center', f'model_{self.idx_model}.pth.tar'),
            torch_device(),
        )
        self.model.eval()

        with torch.no_grad():
            cnn_output, cnn_output_crys = self.model(*self._input_var())
            if self.phys_model != 'moment':
                raise ValueError(f"Unsupported physical model: {self.phys_model}")
            return Moment.moment(self, cnn_output, cnn_output_crys)

    @staticmethod
    def _numpy_bundle(output, parm, zeta, crys_fea):
        """Convert the raw theory-module tensors to NumPy arrays."""
        return (
            output.detach().cpu().numpy(),
            parm.detach().cpu().numpy(),
            zeta.detach().cpu().numpy() ** (1.0 / 7.0),
            crys_fea.detach().cpu().numpy(),
        )

    def predict_d_cen(self, return_all_parm=False, **kwargs):
        """Predict the d-band center from the analytical TinNet theory module."""
        output, parm, zeta, crys_fea = self._predict_moment_outputs()
        if return_all_parm:
            return self._numpy_bundle(output, parm, zeta, crys_fea)
        return output.detach().cpu().numpy()

    def predict_properties(self, return_all_parm=False, **kwargs):
        """Predict filling, center, and full rectangular width."""
        output, parm, zeta, crys_fea = self._predict_moment_outputs()
        if return_all_parm:
            return self._numpy_bundle(output, parm, zeta, crys_fea)
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
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter, crystal_atom_idx, atom_inx):
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
        
        crys_fea = torch.atleast_2d(atom_fea[atom_inx])
        crys_fea = self.conv_to_fc_crys(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc_crys, softplus_crys in zip(self.fcs_crys,
                                              self.softpluses_crys):
                crys_fea = softplus_crys(fc_crys(crys_fea))
        
        out_crys = self.fc_out_crys(crys_fea)
        
        return out, out_crys


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
                 radius=8,
                 dmin=0,
                 step=0.2,
                 dict_atom_fea=None):
        
        # Initialize the class
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
        
    def feas(self,
             image,
             enlarged_image,
             tabulated_nbr_idx,
             tabulated_d_ij,
             enlarged_atom_inx,
             tabulated_v2ds,
             tabulated_v2dd):
        
        n_atoms = len(enlarged_image)
        n_start = len(image) * 60
        n_end = len(image) * 61
        
        enlarged_image = enlarged_image.repeat((3,3,1))
        
        atom_fea = []
        nbr_fea = []
        nbr_fea_idx = []
        idx_syms = []
        nbr_syms = []
        
        shorten_nbr_fea = []
        shorten_atom_fea = []
        shorten_idx_syms = []
        shorten_nbr_fea_idx = []
        shorten_nbr_syms = []
        
        padding_fillter = []
        
        for i in range(0, n_atoms):
            idx = n_atoms*4 + i
            nbr_lists = np.arange(len(enlarged_image))
            nbr_dis = enlarged_image.get_distances(idx, nbr_lists)
            nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
            
            assert len(nbr_lists) <= 86
            
            nbr_dis = nbr_dis[nbr_lists]
            nbr_lists = nbr_lists[np.argsort(nbr_dis)]
            nbr_dis = np.sort(nbr_dis)
            nbr_lists = np.mod(nbr_lists, n_atoms)
            
            nbr_lists = np.pad(nbr_lists,
                               [0, 86-len(nbr_lists)],
                               mode='constant',
                               constant_values=100000)
            
            idx_sym = enlarged_image[idx].symbol
            idx_number = enlarged_image[idx].number
            
            nbr_sym = []
            
            for nbr_list in nbr_lists:
                try:
                    nbr_sym += [enlarged_image[nbr_list].symbol]
                except:
                    nbr_sym += ['XX']
            
            nbr_sym = np.array(nbr_sym)
            
            idx_syms.append(idx_sym)
            nbr_syms.append(nbr_sym)
            
            atom_fea.append(list(self.dict_atom_fea[idx_number]))
            nbr_fea_idx.append(list(nbr_lists))
            
            bond_fea = np.exp(-(nbr_dis[..., np.newaxis]
                                     - self.filter)**2
                                   / self.step**2)
            
            bond_fea = np.pad(bond_fea,
                              [(0, 86-len(bond_fea)), (0, 0)],
                              mode='constant',
                              constant_values=0)
            
            nbr_fea.append(bond_fea)
        
        nbr_fea = np.array(nbr_fea)
        
        tmp = []
        
        for idx_2, nbr_idx in enumerate(nbr_fea_idx[enlarged_atom_inx]):
            if len(np.where(tabulated_nbr_idx == nbr_idx)[0]) != 0:
                tmp += [np.where(tabulated_nbr_idx == nbr_idx)[0]]
            else:
                tmp += [np.array([idx_2])]
        
        tmp = np.concatenate(tmp)
        
        assert len(tmp) == 86
        assert len(set(tmp)) == 86
        
        shorten_nbr_fea.append(nbr_fea[n_start:n_end])
        shorten_atom_fea.append(atom_fea[n_start:n_end])
        shorten_idx_syms.append(idx_syms[n_start:n_end])
        
        nbr_fea_idx = np.array(nbr_fea_idx[n_start:n_end])
        padding_fillter.append(nbr_fea_idx != 100000)
        shorten_nbr_fea_idx.append(np.mod(nbr_fea_idx, 16))
        
        shorten_nbr_syms.append(nbr_syms[n_start:n_end])
        
        shorten_nbr_fea = np.array(shorten_nbr_fea)[0]
        shorten_atom_fea = np.array(shorten_atom_fea)[0]
        shorten_idx_syms = np.array(shorten_idx_syms)[0]
        shorten_nbr_fea_idx = np.array(shorten_nbr_fea_idx)[0]
        shorten_nbr_syms = np.array(shorten_nbr_syms)[0]
        padding_fillter = np.array(padding_fillter)[0]
        
        return (shorten_idx_syms,
                shorten_nbr_syms,
                shorten_atom_fea,
                shorten_nbr_fea,
                shorten_nbr_fea_idx,
                tabulated_d_ij[tmp],
                tabulated_nbr_idx[tmp],
                tabulated_v2dd[tmp],
                tabulated_v2ds[tmp],
                padding_fillter)

