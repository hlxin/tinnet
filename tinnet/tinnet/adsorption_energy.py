#!/usr/bin/env python
"""TinNet adsorption-energy model (Newns-Anderson theory + CGCNN).

The facade is adsorbate-agnostic.  Everything that distinguishes OH from O
(or from any adsorbate you register) lives in an
:class:`~tinnet.tinnet.physics.adsorbates.AdsorbateSpec`:

* which frontier orbitals couple to the d band and how (theory),
* how the adsorbate is placed above the site (structure),
* which pretrained CGCNN ensemble to load (model).

The theory itself is the generic
:class:`~tinnet.tinnet.physics.newns_anderson.NewnsAndersonModel`, used both
for checkpoint inference and for SHAP.
"""
# This script is adapted from Xie's and Ulissi's scripts.

import shap
import torch
import matplotlib.pyplot as plt
import numpy as np

from ase import Atom

try:
    from .band_center import BandCenter
    from .descriptors import VoronoiGraph
    from .gnn import CrystalGraphConvNet
    from .physics import NewnsAndersonModel, AdsorbateSpec, adsorbate_spec, available_adsorbates
    from .tinnet_utils import (
        ADSORBATE_SYMBOLS,
        atom_features,
        adsorption_atom_properties,
        copy_without_adsorbates,
        load_checkpoint_state,
        signed_label,
        signed_color,
        save_figure,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )
except ImportError:  # Allows running this file directly during debugging.
    from band_center import BandCenter
    from descriptors import VoronoiGraph
    from gnn import CrystalGraphConvNet
    from physics import NewnsAndersonModel, AdsorbateSpec, adsorbate_spec, available_adsorbates
    from tinnet_utils import (
        ADSORBATE_SYMBOLS,
        atom_features,
        adsorption_atom_properties,
        copy_without_adsorbates,
        load_checkpoint_state,
        signed_label,
        signed_color,
        save_figure,
        normalize_site_indices,
        pretrained_path,
        set_publication_style,
        torch_device,
        validate_atom_index,
        values_for_symbols,
    )


class AdsorptionEnergy:
    """Predict adsorption energies of any registered adsorbate with TinNet.

    Parameters
    ----------
    image : ase.Atoms
        Slab (adsorbate atoms present in the image are removed automatically).
    site_inx : int or list[int]
        Index of the adsorption site atom (atop: exactly one index).
    adsorbate : str or AdsorbateSpec
        Registered adsorbate name (``'OH'``, ``'O'``, ...) or a spec object.
    name : str
        Label used in printed output and plots.
    """

    def __init__(self, image=None, site_inx=None, adsorbate=None, name='Name'):
        self.image = image
        self.site_inx = site_inx
        self.name = name
        self.atom_fea_dict = atom_features()
        self.atom_prop_dict = adsorption_atom_properties()
        self.descriptor = VoronoiGraph(max_num_nbr=12, radius=8, dmin=0, step=0.2, dict_atom_fea=None)
        self.spec = None
        self.physics = None
        self.phys_model = None
        if adsorbate is not None:
            self.set_adsorbate(adsorbate)

    # ------------------------------------------------------------------
    # Adsorbate handling
    # ------------------------------------------------------------------
    @staticmethod
    def available_adsorbates():
        return available_adsorbates()

    def set_adsorbate(self, adsorbate):
        """Select the adsorbate (name or :class:`AdsorbateSpec`) and its theory module."""
        if isinstance(adsorbate, AdsorbateSpec):
            spec = adsorbate
        else:
            spec = adsorbate_spec(adsorbate)
        self.spec = spec
        self.adsorbate = spec.name
        self.physics = NewnsAndersonModel(spec)
        self.phys_model = spec.phys_model
        return spec

    def _require_spec(self):
        if self.spec is None:
            raise ValueError(
                f"adsorbate must be provided. Supported adsorbates: {', '.join(available_adsorbates())}."
            )
        return self.spec

    # ------------------------------------------------------------------
    # Structure preparation
    # ------------------------------------------------------------------
    def _resolve_prediction_inputs(self, image=None, site_inx=None):
        """Normalize user inputs and return a clean slab plus one atop index."""
        spec = self._require_spec()
        image = self.image if image is None else image
        if image is None:
            raise ValueError("image must be provided.")

        site_inx = self.site_inx if site_inx is None else site_inx
        site_inx = normalize_site_indices(site_inx)
        if len(site_inx) != 1:
            raise NotImplementedError(
                "Only atop adsorption with exactly one site index is currently supported."
            )

        clean_image = copy_without_adsorbates(image, symbols=set(ADSORBATE_SYMBOLS) | set(spec.elements))
        validate_atom_index(clean_image, site_inx[0], label="site_inx")
        return clean_image, site_inx

    @staticmethod
    def build_adsorbed_image(clean_image, site_index, spec):
        """Return a new ASE image with the adsorbate placed above the site."""
        ads_image = clean_image.copy()
        site_position = ads_image.get_positions()[site_index]
        for symbol, offset in spec.geometry:
            ads_image.append(Atom(symbol, position=site_position + np.asarray(offset, dtype=float)))
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
    def _tabulated_d_band(clean_image, site_index):
        """d-band center and semi-ellipse half width from the BandCenter model."""
        band_model = BandCenter(image=clean_image.copy(), atom_inx=site_index)
        d_cen = np.average(band_model.image2band_center(image=clean_image.copy(), atom_inx=site_index))
        full_width = np.average(
            band_model.image2band_full_rectangular_width(image=clean_image.copy(), atom_inx=site_index)
        )
        half_width = full_width / np.sqrt(12) * 2.0
        return d_cen, half_width

    def site_constants(self, clean_image, site_inx):
        """Tabulated constants required by the theory module for this site."""
        spec = self._require_spec()
        constants = {'vad2': self._site_vad2(clean_image, site_inx)}
        if spec.d_band == 'tabulated':
            d_cen, half_width = self._tabulated_d_band(clean_image, site_inx[0])
            constants['d_cen'] = np.array([d_cen], dtype=np.float32)
            constants['width'] = np.array([half_width], dtype=np.float32)
        return constants

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, image=None, site_inx=None, return_all_parm=False):
        """Predict the adsorption energy of the configured adsorbate.

        Returns the ensemble energies ``(n_ensemble,)``; with
        ``return_all_parm=True`` returns ``(n_ensemble, 1 + n_parameters)``
        where column 0 is the energy and the rest follow
        ``self.physics.parameter_names``.
        """
        spec = self._require_spec()
        clean_image, site_inx = self._resolve_prediction_inputs(image, site_inx)
        constants = self.site_constants(clean_image, site_inx)
        ads_image = self.build_adsorbed_image(clean_image, site_inx[0], spec)
        features = self.descriptor.feas(ads_image)

        ensemble = AdsorptionEnsemble(spec, self.physics, features, site_inx)
        model_ead, model_parm = ensemble.predict(constants)

        if return_all_parm:
            return model_parm

        print(
            f"The adsorption energy of {spec.name} on the {spec.site} site "
            f"(index {site_inx}) of {self.name}: "
            f"{np.mean(model_ead):.2f} ± {np.std(model_ead):.2f} eV"
        )
        return model_ead

    def decompose(self, image=None, site_inx=None):
        """Ensemble-averaged theory decomposition (per-orbital hybridization, occupancy, ...)."""
        model_parm = self.predict(image, site_inx, return_all_parm=True)
        with torch.no_grad():
            terms = self.physics.decompose(torch.as_tensor(model_parm[:, 1:], dtype=torch.float32))
        return {
            key: value.numpy().mean(axis=0) if value.dim() > 0 else float(value)
            for key, value in terms.items() if key not in ('ergy', 'dos_d')
        }

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def tinnet_ead(self, parm):
        """Newns-Anderson energy from physical parameters (SHAP model function)."""
        if self.physics is None:
            raise RuntimeError("adsorbate is not set. Run predict() or set_adsorbate() before SHAP analysis.")
        return self.physics.shap_function(parm)

    def _ensemble_shap_matrix(self, predicted_ref, predicted_target):
        """Compute SHAP rows for all ensemble members with one generic loop."""
        shap_rows, ref_energy, target_energy = [], [], []
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

    def gen_shap(self, ref_image, ref_site_inx, target_image, target_site_inx):
        """Return ensemble SHAP contributions and mean parameter differences."""
        predicted_parameter_ref = self.predict(ref_image, ref_site_inx, return_all_parm=True)
        predicted_parameter_target = self.predict(target_image, target_site_inx, return_all_parm=True)
        parm_diff = (np.average(predicted_parameter_target, axis=0)
                     - np.average(predicted_parameter_ref, axis=0))
        return self._ensemble_shap_matrix(predicted_parameter_ref, predicted_parameter_target), parm_diff

    _signed_color = staticmethod(signed_color)
    _signed_label = staticmethod(signed_label)

    def _prepare_shap_waterfall_data(self, shap_values, parm_diff):
        """Sort SHAP features and compute the waterfall start positions."""
        feature_values = np.asarray(shap_values[2:], dtype=float)
        order = np.argsort(feature_values)
        feature_values = feature_values[order]
        parm_values = np.asarray(parm_diff[1:], dtype=float)[order]
        labels = [self.physics.parameter_labels[i] for i in order]

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
            ax.arrow(x=start, y=y, dx=contribution, dy=0, color=color,
                     width=1.0 / 3.0, head_width=1.0 / 3.0,
                     head_length=0.15 * abs(contribution), length_includes_head=True)
            ax.annotate(self._signed_label(contribution), xy=(1.12, y),
                        xycoords=('axes fraction', 'data'), ha='center', va='center', color=color)
            ax.annotate(self._signed_label(parameter_delta), xy=(-0.12, y),
                        xycoords=('axes fraction', 'data'), ha='center', va='center',
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
        ax.set_xlabel(self.physics.property_label)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.tick_params('y', length=0, width=0, which='major')
        ax.annotate(f"{ref_name}\n{start_energy:.2f}", xy=(start_energy, 1.15),
                    xycoords=('data', 'axes fraction'), ha='center', va='center', color='gray')
        ax.annotate(f"{target_name}\n{final_energy:.2f}", xy=(final_energy, 1.05),
                    xycoords=('data', 'axes fraction'), ha='center', va='center', color='orange')
        ax.annotate('SHAP', xy=(1.12, 1.05), xycoords=('axes fraction', 'axes fraction'),
                    ha='center', va='center', color='black')

    def explain_shap(self, ref_image=None, ref_site_inx=None, ref_name='Reference',
                     plot_name='shap', save_fig='png'):
        """Generate a compact SHAP waterfall plot for adsorption energy."""
        if ref_image is None:
            raise ValueError("ref_image must be provided for SHAP explanation.")
        if ref_site_inx is None:
            raise ValueError("ref_site_inx must be provided for SHAP explanation.")
        if self.image is None or self.site_inx is None:
            raise ValueError("target image and site_inx must be stored on the AdsorptionEnergy object.")

        set_publication_style()
        shap_values, parm_diff = self.gen_shap(ref_image, ref_site_inx, self.image, self.site_inx)
        shap_mean = np.average(shap_values, axis=1)
        plot_data = self._prepare_shap_waterfall_data(shap_mean, parm_diff)

        fig, ax = plt.subplots()
        fig.set_size_inches(3.375 * 2.0, 3.375)
        self._draw_shap_waterfall(ax, labels=plot_data[0], feature_values=plot_data[1],
                                  parm_values=plot_data[2], starts=plot_data[3],
                                  y_positions=plot_data[4], start_energy=plot_data[5],
                                  final_energy=plot_data[6], ref_name=ref_name, target_name=self.name)
        fig.tight_layout()
        save_figure(fig, plot_name, save_fig)
        return fig, ax


class AdsorptionEnsemble:
    """Evaluate the pretrained CGCNN ensemble of one adsorbate through its theory module."""

    def __init__(self, spec, physics, features, site_inx, device=None):
        self.spec = spec
        self.physics = physics
        self.device = torch_device() if device is None else device
        atom_fea, nbr_fea, nbr_fea_idx = features
        self.inputs = (
            torch.as_tensor(atom_fea, dtype=torch.float32, device=self.device),
            torch.as_tensor(nbr_fea, dtype=torch.float32, device=self.device),
            torch.as_tensor(nbr_fea_idx, dtype=torch.long, device=self.device),
            [torch.arange(len(atom_fea), dtype=torch.long, device=self.device)],
            torch.as_tensor(site_inx, dtype=torch.long, device=self.device),
        )
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=atom_fea.shape[-1],
            nbr_fea_len=nbr_fea.shape[-1],
            n_out=physics.n_latent,
            readout='mean',
            **spec.gnn,
        ).to(self.device)

    def checkpoint_path(self, model_inx):
        return pretrained_path(*self.spec.checkpoint, f'model_{model_inx}.pth.tar')

    def predict_member(self, model_inx, constants):
        """Return ``(energy (1,), [energy, *parameters] (1 + n_params,))`` for one member."""
        load_checkpoint_state(self.model, self.checkpoint_path(model_inx), self.device)
        self.model.eval()
        with torch.no_grad():
            latent = self.model(*self.inputs)
            constants = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in constants.items()}
            energy, params = self.physics.forward(latent, constants)
        energy = energy.detach().cpu().numpy()
        parm = np.concatenate((energy[:1], params[0].detach().cpu().numpy()))
        return energy, parm

    def predict(self, constants):
        """Stacked ensemble energies ``(n,)`` and parameter rows ``(n, 1 + n_params)``."""
        results = [self.predict_member(i, constants) for i in range(self.spec.n_ensemble)]
        model_ead, model_parm = zip(*results)
        return np.stack(model_ead).flatten(), np.stack(model_parm)


Features = VoronoiGraph  # backward-compatible name
