#!/usr/bin/env python
"""TinNet cohesive-energy prediction for transition-metal alloys.

The renormalized-atom cohesion theory lives in
:class:`~tinnet.tinnet.physics.cohesion.CohesionModel`; this module only
builds the crystal graph, runs the pretrained per-atom CGCNN ensemble through
that theory module, and draws the SHAP waterfall of the four energy terms.
"""
# This script is adapted from scripts of Jeffrey C. Grossman
# and Zachary W. Ulissi.

import numpy as np
import torch
import matplotlib.pyplot as plt
import shap

try:
    from .descriptors import VoronoiGraph
    from .gnn import CrystalGraphConvNet
    from .physics import CohesionModel
    from .tinnet_utils import (
        copy_without_adsorbates,
        cohesive_constant_for_symbols,
        load_checkpoint_state,
        pretrained_path,
        set_publication_style,
        save_figure,
        signed_color,
        signed_label,
        torch_device,
    )
except ImportError:  # Allows running this file directly during debugging.
    from descriptors import VoronoiGraph
    from gnn import CrystalGraphConvNet
    from physics import CohesionModel
    from tinnet_utils import (
        copy_without_adsorbates,
        cohesive_constant_for_symbols,
        load_checkpoint_state,
        pretrained_path,
        set_publication_style,
        save_figure,
        signed_color,
        signed_label,
        torch_device,
    )

Voronoi = VoronoiGraph  # backward-compatible name


class CohesiveEnergy:
    """Predict cohesive energy with the TinNet cohesion-theory module.

    The model returns ensemble predictions and, optionally, interpretable
    contribution terms used by the SHAP explanation routine.
    """

    DEFAULT_GNN = dict(atom_fea_len=150, n_conv=5, h_fea_len=128, n_h=2)
    CHECKPOINT = ('cohesive_energy',)

    def __init__(self, image=None, name='Name'):
        self.image = image
        self.name = name
        self.physics = CohesionModel()
        self.descriptor = VoronoiGraph(max_num_nbr=12, radius=8, dmin=0, step=0.2)

    def site_constants(self, image):
        """Tabulated per-atom constants required by the cohesion theory module."""
        symbols = image.get_chemical_symbols()
        return {
            'promotion_energy': np.asarray(cohesive_constant_for_symbols(symbols, "promotion_energy"), dtype=np.float32),
            'wigner_seitz_volume': np.asarray(cohesive_constant_for_symbols(symbols, "wigner_seitz_volume"), dtype=np.float32),
        }

    def predict(self, image=None, return_all_parm=False, atom_fea_len=150, n_conv=5,
                h_fea_len=128, n_h=2, batch_size=1024, n_ensemble_models=10):
        """Run the cohesive-energy ensemble prediction.

        Returns the ensemble energies ``(n_ensemble,)`` in eV/atom; with
        ``return_all_parm=True`` returns ``(n_ensemble, 5)`` rows of
        ``[E_coh, E_prom, E_renorm, E_s, E_d]`` averaged over atoms.
        """
        del batch_size  # single-structure inference does not need batching
        if image is None:
            image = self.image
        if image is None:
            raise ValueError("image must be provided.")

        clean_image = copy_without_adsorbates(image)
        features = self.descriptor.feas(clean_image)
        constants = self.site_constants(clean_image)
        ensemble = CohesionEnsemble(self.physics, features,
                                    gnn_kwargs=dict(atom_fea_len=atom_fea_len, n_conv=n_conv,
                                                    h_fea_len=h_fea_len, n_h=n_h),
                                    checkpoint=self.CHECKPOINT, n_ensemble=n_ensemble_models)

        ech, parms = [], []
        for member in range(n_ensemble_models):
            site_energy, site_params = ensemble.predict_member(member, constants)
            energy = np.average(site_energy)
            ech.append(energy)
            parms.append(np.concatenate((np.atleast_1d(energy), np.average(site_params, axis=0))))
        ech = np.asarray(ech)
        parms = np.vstack(parms)

        if not return_all_parm:
            print("Cohesive Energy")
            print("=" * 45)
            print(f"{'Material':<15} {'Cohesive Energy (eV/atom)':>20}")
            print("-" * 45)
            print(f"{self.name:<15} {np.average(ech):>20.2f} ± {np.std(ech):<20.2f}")
            return ech
        return parms

    def ech_shap(self, inp_shap):
        """Cohesive energy from its four contribution terms (SHAP model function)."""
        return self.physics.shap_function(inp_shap)

    def gen_shap(self, ref_image, target_image):
        ref_image = copy_without_adsorbates(ref_image)
        target_image = copy_without_adsorbates(target_image)

        predicted_parameter_reference = self.predict(ref_image, return_all_parm=True)
        predicted_parameter_target = self.predict(target_image, return_all_parm=True)

        component_shap_values, predicted_ech_reference, predicted_ech_target = [], [], []
        for ref_params, target_params in zip(predicted_parameter_reference, predicted_parameter_target):
            explainer = shap.Explainer(self.ech_shap, np.atleast_2d(ref_params[1:]))
            shap_values = explainer(np.atleast_2d(target_params[1:])).values
            component_shap_values.append(shap_values.reshape(-1))
            predicted_ech_reference.append(ref_params[0])
            predicted_ech_target.append(target_params[0])

        component_shap_values = np.asarray(component_shap_values)
        return np.vstack((
            np.asarray(predicted_ech_reference).flatten(),
            np.asarray(predicted_ech_target).flatten(),
            component_shap_values.T,
        ))

    def explain_shap(self, ref_image=None, ref_name='Reference', plot_name='shap', save_fig='png'):
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
        labels = np.array(self.physics.parameter_labels)

        order = np.argsort(contributions)
        contributions = contributions[order]
        labels = labels[order]

        starts = base_value + np.r_[0.0, np.cumsum(contributions[:-1])]
        y_positions = np.arange(len(contributions), 0, -1)
        colors = [signed_color(value) for value in contributions]

        for y, start, contribution, color in zip(y_positions, starts, contributions, colors):
            ax.arrow(x=start, y=y, dx=contribution, dy=0, color=color,
                     width=1.0 / 3.0, head_width=1.0 / 3.0,
                     head_length=0.15 * abs(contribution), length_includes_head=True)
            ax.annotate(signed_label(contribution), xy=(-0.12, y), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=color)

        ax.set_ylim([0.5, len(contributions) + 1.5])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(self.physics.property_label)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.tick_params('y', length=0, width=0, which='major')

        ax.plot([base_value, base_value], [len(contributions) + 2, 0.5], '--', color='gray', linewidth=1)
        for start, y in zip(starts[1:], y_positions[:-1]):
            ax.plot([start, start], [y + 1.0 / 3.0, y - 1 - 1.0 / 3.0], '--', color='gray', linewidth=1)

        ax.plot([target_value, target_value], [len(contributions) + 2, 0.5], '--', color='orange', linewidth=1)
        ax.annotate(f"{ref_name}\n{base_value:.2f}", xy=(base_value, 1.15), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='gray')
        ax.annotate(f"{target_name}\n{target_value:.2f}", xy=(target_value, 1.05), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='orange')

        fig.tight_layout()
        save_figure(fig, plot_name, save_fig)
        return fig, ax


class CohesionEnsemble:
    """Evaluate the pretrained per-atom CGCNN ensemble through the cohesion theory module."""

    def __init__(self, physics, features, gnn_kwargs=None, checkpoint=('cohesive_energy',),
                 n_ensemble=10, device=None):
        self.physics = physics
        self.checkpoint = tuple(checkpoint)
        self.n_ensemble = n_ensemble
        self.device = torch_device() if device is None else device
        atom_fea, nbr_fea, nbr_fea_idx = features
        self.inputs = (
            torch.as_tensor(atom_fea, dtype=torch.float32, device=self.device),
            torch.as_tensor(nbr_fea, dtype=torch.float32, device=self.device),
            torch.as_tensor(nbr_fea_idx, dtype=torch.long, device=self.device),
        )
        gnn_kwargs = dict(CohesiveEnergy.DEFAULT_GNN if gnn_kwargs is None else gnn_kwargs)
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=atom_fea.shape[-1],
            nbr_fea_len=nbr_fea.shape[-1],
            n_out=physics.n_latent,
            readout='atom',
            **gnn_kwargs,
        ).to(self.device)

    def checkpoint_path(self, model_inx):
        return pretrained_path(*self.checkpoint, f'model_{model_inx}.pth.tar')

    def predict_member(self, model_inx, constants):
        """Return per-atom ``(energy (N,), parameters (N, 4))`` for one ensemble member."""
        load_checkpoint_state(self.model, self.checkpoint_path(model_inx), self.device)
        self.model.eval()
        with torch.no_grad():
            latent = self.model(*self.inputs)
            constants = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in constants.items()}
            energy, params = self.physics.forward(latent, constants)
        return energy.cpu().numpy(), params.cpu().numpy()
