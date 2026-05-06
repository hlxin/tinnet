#!/usr/bin/env python
"""TinNet adsorption-energy models for OH/O atop adsorption.

This module keeps the original public API while centralizing common utilities,
safer checkpoint loading, and non-mutating ASE structure handling.
"""
# This script is adapted from Xie's and Ulissi's scripts.

import shap
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from ase import Atom
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data.dataloader import default_collate

try:
    from .band_center import BandCenter
except ImportError:  # Allows running this file directly during debugging.
    from band_center import BandCenter

try:
    from .tinnet_utils import (
        atom_features,
        as_float_tensor,
        adsorption_atom_properties,
        as_long_tensor,
        copy_without_adsorbates,
        data_path,
        load_checkpoint_state,
        signed_label,
        signed_color,
        save_figure,
        move_tensor_sequence,
        make_data_loader,
        normalize_atom_index,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        stack_scalar_parameters,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )
except ImportError:  # Allows running this file directly during debugging.
    from tinnet_utils import (
        atom_features,
        as_float_tensor,
        adsorption_atom_properties,
        as_long_tensor,
        copy_without_adsorbates,
        data_path,
        load_checkpoint_state,
        signed_label,
        signed_color,
        save_figure,
        move_tensor_sequence,
        make_data_loader,
        normalize_atom_index,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        stack_scalar_parameters,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )


class AdsorptionEnergy:
    """Predict OH/O atop adsorption energies with TinNet.

    The implementation is configuration driven: adsorbate-specific metadata
    (parameter names, SHAP labels, orbital degeneracies, and plot labels) lives
    in one place and is reused by prediction, SHAP analysis, the Newns-Anderson
    post-hoc energy function, and plotting. This avoids separate OH/O code paths
    that can silently diverge over time.
    """

    N_ENSEMBLE_MODELS = 10

    MODEL_SPECS = {
        'OH': {
            'phys_model': 'OH_atop',
            'adsorbate_geometry': (
                ('O', np.array([0.0, 0.0, 2.00])),
                ('H', np.array([0.8, 0.0, 2.41])),
            ),
            'energy_label': r'$E_{ad}^{OH, top}$ (eV)',
            'feature_labels': [
                r'$V_{ad}^{2}$',
                r'$\epsilon_{d}$',
                r'$W_{d}$',
                r'$\epsilon_{3\sigma}$',
                r'$\beta_{3\sigma}$',
                r'$\delta_{3\sigma}$',
                r'$\epsilon_{1\pi}$',
                r'$\beta_{1\pi}$',
                r'$\delta_{1\pi}$',
                r'$\epsilon_{4\sigma^{*}}$',
                r'$\beta_{4\sigma^{*}}$',
                r'$\delta_{4\sigma^{*}}$',
            ],
            # Columns in return_all_parm after the first energy column.
            'parameter_columns': (
                'vad2', 'd_cen', 'width',
                'adse_1', 'beta_1', 'delta_1',
                'adse_2', 'beta_2', 'delta_2',
                'adse_3', 'beta_3', 'delta_3',
            ),
            # Adsorbate orbital blocks in the parameter matrix.
            # (adse column, beta column, delta column, degeneracy, alpha)
            'orbital_blocks': (
                (3, 4, 5, 1, 0.06378761202273762),
                (6, 7, 8, 2, 0.06378761202273762),
                (9, 10, 11, 1, 0.06378761202273762),
            ),
            'esp': -2.693696878597913,
        },
        'O': {
            'phys_model': 'O_atop',
            'adsorbate_geometry': (
                ('O', np.array([0.0, 0.0, 1.80])),
            ),
            'energy_label': r'$E_{ad}^{O, top}$ (eV)',
            'feature_labels': [
                r'$V_{ad}^{2}$',
                r'$\epsilon_{d}$',
                r'$W_{d}$',
                r'$\epsilon_{pz}$',
                r'$\beta_{pz}$',
                r'$\delta_{pz}$',
                r'$\epsilon_{pxy}$',
                r'$\beta_{pxy}$',
                r'$\delta_{pxy}$',
            ],
            'parameter_columns': (
                'vad2', 'd_cen', 'width',
                'adse_2', 'beta_2', 'delta_2',
                'adse_3', 'beta_3', 'delta_3',
            ),
            'orbital_blocks': (
                (3, 4, 5, 1, 0.07889181751783157),
                (6, 7, 8, 2, 0.05687347683456299),
            ),
            'esp': -3.765294246337454,
        },
    }

    def __init__(self,
                 image=None,
                 site_inx=None,
                 adsorbate=None,
                 name='Name'):
        self.image = image
        self.site_inx = site_inx
        self.adsorbate = adsorbate
        self.name = name
        self.atom_fea_dict = atom_features()
        self.atom_prop_dict = adsorption_atom_properties()
        self.descriptor = Features(max_num_nbr=12,
                                   radius=8,
                                   dmin=0,
                                   step=0.2,
                                   dict_atom_fea=None)
        self.phys_model = None

    @classmethod
    def _spec_for(cls, adsorbate=None, phys_model=None):
        """Return adsorbate metadata from either adsorbate name or physical-model name."""
        if adsorbate is None and phys_model is not None:
            for candidate, spec in cls.MODEL_SPECS.items():
                if spec['phys_model'] == phys_model:
                    return candidate, spec
            raise ValueError(f"Unsupported physical model: {phys_model}")

        if adsorbate not in cls.MODEL_SPECS:
            supported = ', '.join(sorted(cls.MODEL_SPECS))
            raise ValueError(f"Unsupported adsorbate '{adsorbate}'. Supported adsorbates: {supported}.")
        return adsorbate, cls.MODEL_SPECS[adsorbate]

    def _spec(self, adsorbate=None, phys_model=None):
        """Return adsorbate metadata for this predictor instance."""
        adsorbate = self.adsorbate if adsorbate is None and phys_model is None else adsorbate
        return self._spec_for(adsorbate=adsorbate, phys_model=phys_model)

    def _resolve_prediction_inputs(self, image=None, site_inx=None):
        """Normalize user inputs and return a clean slab plus one atop index."""
        image = self.image if image is None else image
        if image is None:
            raise ValueError("image must be provided.")

        site_inx = self.site_inx if site_inx is None else site_inx
        site_inx = normalize_site_indices(site_inx)
        if len(site_inx) != 1:
            raise NotImplementedError(
                "Only atop adsorption with exactly one site index is currently supported."
            )

        clean_image = copy_without_adsorbates(image)
        validate_atom_index(clean_image, site_inx[0], label="site_inx")
        return clean_image, site_inx

    @staticmethod
    def _build_adsorbed_image(clean_image, site_index, adsorbate_geometry):
        """Return a new ASE image with the adsorbate placed above the atop site."""
        ads_image = clean_image.copy()
        site_position = ads_image.get_positions()[site_index]
        for symbol, offset in adsorbate_geometry:
            ads_image.append(Atom(symbol, position=site_position + offset))
        return ads_image

    def _site_vad2(self, clean_image, site_inx):
        symbols = clean_image.get_chemical_symbols()
        return np.array(
            values_for_symbols([symbols[i] for i in site_inx],
                               {k: v['vad2'] for k, v in self.atom_prop_dict.items()},
                               'vad2'),
            dtype=np.float32,
        )

    @staticmethod
    def _band_parameters_for_o_atop(clean_image, site_index):
        """Compute d-band center and rectangular half-width required by O-atop."""
        band_model = BandCenter(image=clean_image.copy(), atom_inx=site_index)
        d_cen = np.average(band_model.image2band_center(image=clean_image.copy(),
                                                        atom_inx=site_index))
        full_width = np.average(
            band_model.image2band_full_rectangular_width(image=clean_image.copy(),
                                                         atom_inx=site_index)
        )
        half_width = full_width / np.sqrt(12) * 2.0
        return d_cen, half_width

    def _run_ensemble(self, *, features, phys_model, site_inx, vad2, model_kwargs=None):
        """Evaluate all pretrained ensemble members and return stacked outputs."""
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        results = []
        for model_inx in range(self.N_ENSEMBLE_MODELS):
            model = Regression(features=features,
                               model_inx=model_inx,
                               vad2=vad2,
                               site_inx=site_inx,
                               phys_model=phys_model,
                               **model_kwargs)
            results.append(model.eval_model())

        model_ead, model_parm = zip(*results)
        return np.stack(model_ead).flatten(), np.stack(model_parm)

    def predict(self,
                image=None,
                site_inx=None,
                return_all_parm=False):
        """Predict adsorption energy for the configured atop adsorbate."""
        adsorbate, spec = self._spec()
        clean_image, site_inx = self._resolve_prediction_inputs(image, site_inx)
        site_index = site_inx[0]
        self.phys_model = spec['phys_model']

        model_kwargs = {}
        if adsorbate == 'O':
            d_cen, half_width = self._band_parameters_for_o_atop(clean_image, site_index)
            model_kwargs.update(d_cen=d_cen, half_width=half_width)

        ads_image = self._build_adsorbed_image(clean_image,
                                               site_index,
                                               spec['adsorbate_geometry'])
        features = self.descriptor.feas(ads_image)
        vad2 = self._site_vad2(clean_image, site_inx)

        model_ead, model_parm = self._run_ensemble(features=features,
                                                   phys_model=spec['phys_model'],
                                                   site_inx=site_inx,
                                                   vad2=vad2,
                                                   model_kwargs=model_kwargs)

        if return_all_parm:
            return model_parm

        print(
            f"The adsorption energy of {adsorbate} on the atop site "
            f"(index {site_inx}) of {self.name}: "
            f"{np.mean(model_ead):.2f} ± {np.std(model_ead):.2f} eV"
        )
        return model_ead

    def _ensemble_shap_matrix(self, predicted_ref, predicted_target):
        """Compute SHAP rows for all ensemble members with one generic loop."""
        shap_rows = []
        ref_energy = []
        target_energy = []

        for ref_params, target_params in zip(predicted_ref, predicted_target):
            ref_params = np.asarray(ref_params)
            target_params = np.asarray(target_params)
            explainer = shap.Explainer(self.tinnet_ead, np.atleast_2d(ref_params[1:]))
            shap_values = explainer(np.atleast_2d(target_params[1:])).values

            ref_energy.append(ref_params[0])
            target_energy.append(target_params[0])
            shap_rows.append(np.asarray(shap_values).reshape(-1))

        return np.vstack((
            np.asarray(ref_energy).reshape(1, -1),
            np.asarray(target_energy).reshape(1, -1),
            np.asarray(shap_rows).T,
        ))

    def gen_shap(self,
                 ref_image,
                 ref_site_inx,
                 target_image,
                 target_site_inx):
        """Return ensemble SHAP contributions and mean parameter differences."""
        predicted_parameter_ref = self.predict(ref_image,
                                               ref_site_inx,
                                               return_all_parm=True)
        predicted_parameter_target = self.predict(target_image,
                                                  target_site_inx,
                                                  return_all_parm=True)
        parm_diff = (np.average(predicted_parameter_target, axis=0)
                     - np.average(predicted_parameter_ref, axis=0))
        return self._ensemble_shap_matrix(predicted_parameter_ref,
                                          predicted_parameter_target), parm_diff

    @staticmethod
    def _hilbert_multiplier(n_grid, *, dtype, device):
        """Return the FFT multiplier used for the Hilbert transform."""
        h = torch.zeros(n_grid, dtype=dtype, device=device)
        if n_grid % 2 == 0:
            h[0] = 1
            h[n_grid // 2] = 1
            h[1:n_grid // 2] = 2
        else:
            h[0] = 1
            h[1:(n_grid + 1) // 2] = 2
        return h

    @staticmethod
    def _safe_denominator(value, eps):
        """Avoid singular denominators while preserving the original sign rule."""
        small = torch.abs(value) <= eps
        return (value * (~small)
                + eps * small * (value >= 0)
                - eps * small * (value < 0))

    @staticmethod
    def _semi_ellipse_dos_batch(ergy, d_cen, width):
        """Build a normalized semi-elliptic d-DOS for a batch of sites."""
        dos_d = torch.abs(1 - ((ergy[None, :] - d_cen[:, None]) / width[:, None]) ** 2) ** 0.5
        dos_d = dos_d * (torch.abs(ergy[None, :] - d_cen[:, None]) < width[:, None])
        area = torch.trapz(dos_d, ergy, dim=1)
        dos_d = dos_d + (area[:, None] <= 1e-10) / len(ergy)
        return dos_d / torch.trapz(dos_d, ergy, dim=1)[:, None]

    def _newns_orbital_terms_batch(self, *, ergy, h, fermi, vad2, dos_d,
                                   adse, beta, delta, eps):
        """Evaluate one adsorbate frontier orbital for a batch of parameter sets."""
        wdos = np.pi * (beta[:, None] * vad2[:, None] * dos_d) + delta[:, None]
        wdos_reference = np.pi * (0 * vad2[:, None] * dos_d) + delta[:, None]

        htwdos = torch.imag(torch.fft.ifft(torch.fft.fft(wdos, dim=1) * h[None, :], dim=1))
        denominator = self._safe_denominator(ergy[None, :] - adse[:, None] - htwdos, eps)
        arctan = torch.atan(wdos / denominator)
        arctan = (arctan - np.pi) * (arctan > 0) + arctan * (arctan <= 0)
        d_hyb = 2 / np.pi * torch.trapz(arctan[:, :fermi], ergy[:fermi], dim=1)

        lorentzian = (1 / np.pi) * delta[:, None] / ((ergy[None, :] - adse[:, None]) ** 2 + delta[:, None] ** 2)
        na = torch.trapz(lorentzian[:, :fermi], ergy[:fermi], dim=1)

        denominator_ref = self._safe_denominator(ergy[None, :] - adse[:, None], eps)
        arctan_ref = torch.atan(wdos_reference / denominator_ref)
        arctan_ref = ((arctan_ref - np.pi) * (arctan_ref > 0)
                      + arctan_ref * (arctan_ref <= 0))
        d_hyb_ref = 2 / np.pi * torch.trapz(arctan_ref[:, :fermi], ergy[:fermi], dim=1)
        return d_hyb - d_hyb_ref, na

    def tinnet_ead(self, parm):
        """Vectorized Newns-Anderson adsorption-energy function for SHAP."""
        if self.phys_model is None:
            raise RuntimeError("phys_model is not set. Run predict() before SHAP analysis.")
        _, spec = self._spec(phys_model=self.phys_model)

        parm = torch.as_tensor(parm, dtype=torch.float32)
        ergy = torch.linspace(-15, 15, 3001, dtype=parm.dtype, device=parm.device)
        h = self._hilbert_multiplier(len(ergy), dtype=parm.dtype, device=parm.device)
        fermi = int(torch.argmin(torch.abs(ergy)).item()) + 1
        eps = np.finfo(float).eps

        vad2 = parm[:, 0]
        d_cen = parm[:, 1]
        width = parm[:, 2]
        dos_d = self._semi_ellipse_dos_batch(ergy, d_cen, width)
        filling = torch.trapz(dos_d[:, :fermi], ergy[:fermi], dim=1)

        energy = torch.full_like(vad2, float(spec['esp']))
        for adse_col, beta_col, delta_col, degeneracy, alpha in spec['orbital_blocks']:
            energy_na, occupancy = self._newns_orbital_terms_batch(
                ergy=ergy,
                h=h,
                fermi=fermi,
                vad2=vad2,
                dos_d=dos_d,
                adse=parm[:, adse_col],
                beta=parm[:, beta_col],
                delta=parm[:, delta_col],
                eps=eps,
            )
            energy = energy + degeneracy * (energy_na + 2 * (occupancy + filling) * alpha * parm[:, beta_col] * vad2)

        return np.atleast_1d(energy.detach().cpu().numpy())

    _signed_color = staticmethod(signed_color)
    _signed_label = staticmethod(signed_label)

    def _prepare_shap_waterfall_data(self, shap_values, parm_diff):
        """Sort SHAP features and compute the waterfall start positions."""
        feature_values = np.asarray(shap_values[2:], dtype=float)
        order = np.argsort(feature_values)
        feature_values = feature_values[order]
        parm_values = np.asarray(parm_diff[1:], dtype=float)[order]
        labels = [self._spec(phys_model=self.phys_model)[1]['feature_labels'][i] for i in order]

        start_energy = float(shap_values[0])
        end_energy = float(shap_values[1])
        starts = start_energy + np.r_[0.0, np.cumsum(feature_values[:-1])]
        final_energy = starts[-1] + feature_values[-1]
        y_positions = np.arange(len(feature_values), 0, -1)
        return labels, feature_values, parm_values, starts, y_positions, start_energy, final_energy, end_energy

    def _draw_shap_waterfall(self, ax, *, labels, feature_values, parm_values,
                             starts, y_positions, start_energy, final_energy,
                             ref_name, target_name):
        """Draw the adsorption-energy SHAP waterfall with one loop."""
        for start, contribution, parameter_delta, y in zip(starts, feature_values, parm_values, y_positions):
            color = self._signed_color(contribution)
            ax.arrow(x=start,
                     y=y,
                     dx=contribution,
                     dy=0,
                     color=color,
                     width=1.0 / 3.0,
                     head_width=1.0 / 3.0,
                     head_length=0.15 * abs(contribution),
                     length_includes_head=True)
            ax.annotate(self._signed_label(contribution),
                        xy=(1.12, y),
                        xycoords=('axes fraction', 'data'),
                        ha='center',
                        va='center',
                        color=color)
            ax.annotate(self._signed_label(parameter_delta),
                        xy=(-0.12, y),
                        xycoords=('axes fraction', 'data'),
                        ha='center',
                        va='center',
                        color=self._signed_color(parameter_delta))

        for i, x in enumerate(starts):
            if i == 0:
                ax.plot([x, x], [len(y_positions) + 1, 0.5], '--', color='gray', linewidth=1)
            else:
                y_upper = y_positions[i - 1] - 1.0 / 3.0
                y_lower = y_positions[i] + 1.0 / 3.0
                ax.plot([x, x], [y_lower, y_upper], '--', color='gray', linewidth=1)

        ax.plot([final_energy, final_energy], [len(y_positions) + 1, 0.5], '--', color='orange', linewidth=1)
        ax.set_ylim([0.5, len(y_positions) + 0.5])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(self._spec(phys_model=self.phys_model)[1]['energy_label'])
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.tick_params('y', length=0, width=0, which='major')
        ax.annotate(f"{ref_name}\n{start_energy:.2f}",
                    xy=(start_energy, 1.15),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='center',
                    color='gray')
        ax.annotate(f"{target_name}\n{final_energy:.2f}",
                    xy=(final_energy, 1.05),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='center',
                    color='orange')
        ax.annotate('SHAP',
                    xy=(1.12, 1.05),
                    xycoords=('axes fraction', 'axes fraction'),
                    ha='center',
                    va='center',
                    color='black')

    def explain_shap(self,
                     ref_image=None,
                     ref_site_inx=None,
                     ref_name='Reference',
                     plot_name='shap',
                     save_fig='png'):
        """Generate a compact SHAP waterfall plot for adsorption energy."""
        if ref_image is None:
            raise ValueError("ref_image must be provided for SHAP explanation.")
        if ref_site_inx is None:
            raise ValueError("ref_site_inx must be provided for SHAP explanation.")
        if self.image is None or self.site_inx is None:
            raise ValueError("target image and site_inx must be stored on the AdsorptionEnergy object.")

        target_image = self.image
        target_site_inx = self.site_inx
        target_name = self.name
        set_publication_style()

        shap_values, parm_diff = self.gen_shap(ref_image,
                                               ref_site_inx,
                                               target_image,
                                               target_site_inx)
        shap_mean = np.average(shap_values, axis=1)
        plot_data = self._prepare_shap_waterfall_data(shap_mean, parm_diff)

        fig, ax = plt.subplots()
        fig.set_size_inches(3.375 * 2.0, 3.375)
        self._draw_shap_waterfall(ax,
                                  labels=plot_data[0],
                                  feature_values=plot_data[1],
                                  parm_values=plot_data[2],
                                  starts=plot_data[3],
                                  y_positions=plot_data[4],
                                  start_energy=plot_data[5],
                                  final_energy=plot_data[6],
                                  ref_name=ref_name,
                                  target_name=target_name)
        fig.tight_layout()

        save_figure(fig, plot_name, save_fig)
        return fig, ax

class Regression:
    def __init__(self,
                 features,
                 name_images=None,
                 phys_model=None,
                 model_inx=None,
                 batch_size=256,
                 num_workers=0,
                 vad2=None,
                 site_inx=None,
                 d_cen=None,
                 half_width=None,
                 **kwargs
                 ):
        
        # Initialize Physical Model
        Chemisorption.__init__(self, phys_model, **kwargs)
        
        atom_fea, nbr_fea, nbr_fea_idx = features
        
        # h for Hilbert Transform
        h = np.zeros(3001)
        if 3001 % 2 == 0:
            h[0] = h[3001 // 2] = 1
            h[1:3001 // 2] = 2
        else:
            h[0] = 1
            h[1:(3001+1) // 2] = 2
        
        ergy = np.linspace(-15, 15, 3001)
        
        if name_images is None:
            name_images = np.arange(len(atom_fea))
        
        dataset = [((torch.as_tensor(atom_fea, dtype=torch.float32),
                     torch.as_tensor(nbr_fea, dtype=torch.float32),
                     torch.LongTensor(nbr_fea_idx)),
                    name_images,
                    site_inx)]
        
        cuda = torch.cuda.is_available()
        
        collate_fn = self.collate_pool
        
        data_loader = self.get_data_loader(dataset=dataset,
                                           collate_fn=collate_fn,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=cuda)
        if phys_model == 'OH_atop':
            # build model
            model = CrystalGraphConvNet(orig_atom_fea_len=atom_fea.shape[-1],
                                        nbr_fea_len=nbr_fea.shape[-1],
                                        atom_fea_len=95,
                                        n_conv=5,
                                        h_fea_len=174,
                                        n_h=2,
                                        model_num_input=self.model_num_input)
            self.esp = -2.693696878597913
        if phys_model == 'O_atop':
            model = CrystalGraphConvNet(orig_atom_fea_len=atom_fea.shape[-1],
                                        nbr_fea_len=nbr_fea.shape[-1],
                                        atom_fea_len=71,
                                        n_conv=7,
                                        h_fea_len=104,
                                        n_h=4,
                                        model_num_input=self.model_num_input)
            self.esp = -3.765294246337454
            self.d_cen = d_cen
            self.half_width = half_width
        
        if cuda:
            model.cuda()
        
        # Initialize the class
        if cuda:
            h = torch.FloatTensor(h).cuda()
            ergy = torch.FloatTensor(ergy).cuda()
            vad2 = torch.from_numpy(vad2).cuda()
        else:
            h = torch.FloatTensor(h)
            ergy = torch.FloatTensor(ergy)
            vad2 = torch.from_numpy(vad2)
        
        self.cuda = cuda
        self.data_loader = data_loader
        self.ergy = ergy
        self.h = h
        self.model_inx = model_inx
        self.model = model
        self.phys_model = phys_model
        self.site_inx = site_inx
        self.vad2 = vad2
    
    def _checkpoint_path(self):
        """Return the pretrained checkpoint for the current adsorption model."""
        adsorbate, spec = AdsorptionEnergy._spec_for(phys_model=self.phys_model)
        return pretrained_path('adsorption_energy', adsorbate, 'atop', f'model_{self.model_inx}.pth.tar')

    def _move_input_batch(self, batch_input):
        """Move a collated adsorption graph batch to the active device."""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, site_ids = batch_input
        device = torch_device()
        return (
            atom_fea.to(device, non_blocking=True),
            nbr_fea.to(device, non_blocking=True),
            nbr_fea_idx.to(device, non_blocking=True),
            [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],
            [idx.to(device, non_blocking=True) for idx in site_ids],
        )

    def eval_model(self, **kwargs):
        """Evaluate one pretrained adsorption-energy ensemble member."""
        load_checkpoint_state(self.model, self._checkpoint_path(), torch_device())
        self.model.eval()

        output = parm = None
        with torch.no_grad():
            for batch_input, batch_cif_ids in self.data_loader:
                input_var = self._move_input_batch(batch_input)
                cnn_output = self.model(*input_var)
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    phys_model=self.phys_model,
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids),
                )

        if output is None or parm is None:
            raise RuntimeError("No adsorption prediction was produced; check the data loader.")
        return output, parm

    def get_data_loader(self,
                        dataset,
                        collate_fn=default_collate,
                        batch_size=256,
                        num_workers=0,
                        pin_memory=False,
                        random_seed=None):
        """Return a one-pass inference loader for one adsorption structure."""
        return make_data_loader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def collate_pool(self, dataset_list):
        '''
        Collate a list of data and return a batch for predicting crystal
        properties.
    
        Parameters
        ----------
    
        dataset_list: list of tuples for each data point.
          (atom_fea, nbr_fea, nbr_fea_idx)
    
          atom_fea: torch.Tensor shape (n_i, atom_fea_len)
          nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
          nbr_fea_idx: torch.LongTensor shape (n_i, M)
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
        batch_cif_ids: list
        '''
        batch_atom_fea = []
        batch_nbr_fea = []
        batch_nbr_fea_idx = []
        crystal_atom_idx = []
        batch_cif_ids = []
        batch_site_ids = []
        base_idx = 0
        for i, ((atom_fea, nbr_fea, nbr_fea_idx), cif_id, site_id)\
                in enumerate(dataset_list):
            n_i = atom_fea.shape[0]  # number of atoms for this crystal
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
            new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
            crystal_atom_idx.append(new_idx)
            batch_cif_ids.append(cif_id)
            batch_site_ids.append(site_id)
            base_idx += n_i
        return (torch.cat(batch_atom_fea, dim=0),
                torch.cat(batch_nbr_fea, dim=0),
                torch.cat(batch_nbr_fea_idx, dim=0),
                crystal_atom_idx,
                torch.LongTensor(batch_site_ids)),\
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
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, site_inx):
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

        atom_fea: (torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        '''
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)




def _safe_newns_denominator(denominator: torch.Tensor, eps: float) -> torch.Tensor:
    """Avoid singular denominators while preserving the original sign convention."""
    abs_denom = torch.abs(denominator)
    return (denominator * (abs_denom > eps)
            + eps * (abs_denom <= eps) * (denominator >= 0)
            - eps * (abs_denom <= eps) * (denominator < 0))


def _semi_ellipse_dos(ergy: torch.Tensor, d_cen: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """Build the normalized semi-elliptical d-band DOS used by the TinNet theory module."""
    dos_d = (torch.abs(1 - ((ergy - d_cen) / width) ** 2)) ** 0.5
    dos_d = dos_d * (torch.abs(ergy - d_cen) < width)
    dos_d = dos_d + (torch.trapz(dos_d, ergy) <= 1e-10) / len(ergy)
    return dos_d / torch.trapz(dos_d, ergy)


def _newns_orbital_terms(
    *,
    ergy: torch.Tensor,
    h: torch.Tensor,
    fermi: int,
    vad2: torch.Tensor,
    dos_d: torch.Tensor,
    adse: torch.Tensor,
    beta: torch.Tensor,
    delta: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Newns-Anderson hybridization term and free adsorbate occupancy.

    This routine is shared by all adsorbate frontier orbitals. It replaces the
    repeated orbital-specific blocks in the original implementation without
    changing the mathematical operations.
    """
    wdos = np.pi * (beta * vad2 * dos_d) + delta
    wdos_reference = np.pi * (0 * vad2 * dos_d) + delta

    htwdos = torch.imag(torch.fft.ifft(torch.fft.fft(wdos) * h))
    denominator = _safe_newns_denominator(ergy - adse - htwdos, eps)
    arctan = torch.atan(wdos / denominator)
    arctan = (arctan - np.pi) * (arctan > 0) + arctan * (arctan <= 0)
    d_hyb = 2 / np.pi * torch.trapz(arctan[:fermi], ergy[:fermi])

    lorentzian = (1 / np.pi) * delta / ((ergy - adse) ** 2 + delta ** 2)
    na = torch.trapz(lorentzian[:fermi], ergy[:fermi])

    denominator_reference = _safe_newns_denominator(ergy - adse, eps)
    arctan_reference = torch.atan(wdos_reference / denominator_reference)
    arctan_reference = ((arctan_reference - np.pi) * (arctan_reference > 0)
                        + arctan_reference * (arctan_reference <= 0))
    d_hyb_reference = 2 / np.pi * torch.trapz(arctan_reference[:fermi], ergy[:fermi])

    return d_hyb - d_hyb_reference, na


class Chemisorption:
    """Theory module used during checkpoint inference.

    This class uses the same orbital-loop implementation pattern as
    AdsorptionEnergy.tinnet_ead, but keeps scalar tensor operations because
    the pretrained Regression model evaluates one structure at a time.
    """
    def __init__(self, phys_model, **kwargs):
        # Initialize the class
        if phys_model == 'OH_atop':
            self.alpha = 0.06378761202273762
            self.model_num_input = 11
        if phys_model == 'O_atop':
            self.alpha_2 = 0.07889181751783157
            self.alpha_3 = 0.05687347683456299
            self.model_num_input = 6
    
    def newns_anderson_semi(self, namodel_in, phys_model, **kwargs):
        """Evaluate the semi-elliptical Newns-Anderson theory module.

        The original implementation repeated the same Hilbert-transform and
        hybridization-energy calculation for each adsorbate frontier orbital.  This
        version expresses each orbital as a compact specification while preserving
        the exact ordering of returned parameters used by SHAP and the notebook.
        """
        namodel_in = torch.flatten(namodel_in)
        vad2 = self.vad2
        h = self.h
        ergy = self.ergy
        fermi = np.argsort(torch.abs(ergy).detach().cpu().numpy())[0] + 1
        eps = np.finfo(float).eps

        if phys_model == 'OH_atop':
            d_cen = namodel_in[9]
            width = torch.nn.functional.softplus(namodel_in[10])
            orbital_specs = [
                # (adsorbate resonance, coupling coefficient, broadening, degeneracy, orthogonalization coefficient)
                (namodel_in[0], torch.nn.functional.softplus(namodel_in[1]), namodel_in[2], 1, self.alpha),
                (namodel_in[3], torch.nn.functional.softplus(namodel_in[4]), namodel_in[5], 2, self.alpha),
                (namodel_in[6], torch.nn.functional.softplus(namodel_in[7]), namodel_in[8], 1, self.alpha),
            ]
        elif phys_model == 'O_atop':
            d_cen = self.d_cen
            width = self.half_width
            orbital_specs = [
                (namodel_in[0], torch.nn.functional.softplus(namodel_in[1]), namodel_in[2], 1, self.alpha_2),
                (namodel_in[3], torch.nn.functional.softplus(namodel_in[4]), namodel_in[5], 2, self.alpha_3),
            ]
        else:
            raise ValueError(f"Unsupported chemisorption model: {phys_model}")

        dos_d = _semi_ellipse_dos(ergy, d_cen, width)
        filling = torch.trapz(dos_d[:fermi], ergy[:fermi])

        energy = self.esp
        returned_parameters = [energy.new_tensor(0.0) if torch.is_tensor(energy) else torch.as_tensor(0.0, dtype=ergy.dtype, device=ergy.device),
                               vad2[0], d_cen, width]

        orbital_parameters = []
        for adse, beta, delta_raw, degeneracy, alpha in orbital_specs:
            delta = torch.nn.functional.softplus(delta_raw)
            energy_na, adsorbate_occupancy = _newns_orbital_terms(
                ergy=ergy,
                h=h,
                fermi=fermi,
                vad2=vad2,
                dos_d=dos_d,
                adse=adse,
                beta=beta,
                delta=delta,
                eps=eps,
            )
            energy = energy + degeneracy * (energy_na + 2 * (adsorbate_occupancy + filling) * alpha * beta * vad2)
            orbital_parameters.extend((adse, beta, delta))

        returned_parameters[0] = energy[0]
        parm = stack_scalar_parameters(tuple(returned_parameters + orbital_parameters), like=energy)
        return energy.detach().cpu().numpy(), parm

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

