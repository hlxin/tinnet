#!/usr/bin/env python
# This script is adapted from Xie's and Ulissi's scripts.

import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap
import torch
import torch.nn as nn

from torch.autograd import Variable
from ase import io
from pylab import *


def _copy_without_adsorbates(image):
    """Return a copy of an ASE Atoms object with O/H adsorbates removed.

    This avoids mutating the caller's structure during prediction or SHAP analysis.
    """
    clean_image = image.copy()
    adsorbate_indices = [
        i for i, atom in enumerate(clean_image) if atom.symbol in {"O", "H"}
    ]
    for i in sorted(adsorbate_indices, reverse=True):
        del clean_image[i]
    return clean_image


class BandCenter:
    def __init__(self,
                 image=None,
                 atom_inx=None,
                 name='Name'):
        self.descriptor = Features(radius=8,
                                   dmin=0,
                                   step=0.2,
                                   dict_atom_fea=None)
        self.material_dict = Features.material_dict()
        self.image = image
        self.atom_inx = atom_inx
        self.name = name
    
    def predict(self,
                return_all_parm=False):
        
        if self.image is None:
            raise ValueError("image must be provided.")
        if self.atom_inx is None:
            raise ValueError("atom_inx must be provided.")

        input_image = _copy_without_adsorbates(self.image)
        
        atom_inx = self.atom_inx
        system_name = self.name
        if atom_inx + 1 > len(input_image):
            raise IndexError(f"atom_inx={atom_inx} is outside the clean image with {len(input_image)} atoms.")
        
        enlarged_image = input_image.copy()
        enlarged_atom_inx = len(enlarged_image)*60 + atom_inx
        enlarged_image = enlarged_image.repeat((11,11,1))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        tabulated_filling_inf = self.material_dict[idx_sym]['bulk_filling']
        tabulated_d_cen_inf = self.material_dict[idx_sym]['d_cen']
        tabulated_full_width_inf = self.material_dict[idx_sym]['full_width']
        
        idx_rad = self.material_dict[idx_sym]['rd']
        nbr_rad = np.array([self.material_dict[nbr_sym]['rd'] for nbr_sym in nbr_syms])
        
        vds = idx_rad**1.5 / nbr_dis**3.5
        vdd = idx_rad**1.5 * nbr_rad**1.5 / nbr_dis**5.0
        
        v2ds = 9.9856 * vds**2.0 * 7.62**2
        v2dd = 415.565 * vdd**2.0 * 7.62**2
        
        assert len(v2ds) <= 86
        assert len(v2dd) <= 86
        assert len(nbr_dis) <= 86
        assert len(nbr_lists) <= 86
        
        tabulated_v2ds = np.pad(v2ds, [0, 86-len(v2ds)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_v2dd = np.pad(v2dd, [0, 86-len(v2dd)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_d_ij = np.pad(nbr_dis, [0, 86-len(nbr_dis)],
                                mode='constant',
                                constant_values=1E6)
        
        tabulated_nbr_idx = np.pad(nbr_lists, [0, 86-len(nbr_lists)],
                                   mode='constant',
                                   constant_values=1E6)
        
        (shorten_idx_syms,
         shorten_nbr_syms,
         shorten_atom_fea,
         shorten_nbr_fea,
         shorten_nbr_fea_idx,
         tabulated_d_ij_sorted,
         tabulated_nbr_index_sorted,
         tabulated_v2dd_sorted,
         tabulated_v2ds_sorted,
         tabulated_padding_fillter) = self.descriptor.feas(input_image,
                                                           enlarged_image,
                                                           tabulated_nbr_idx,
                                                           tabulated_d_ij,
                                                           enlarged_atom_inx,
                                                           tabulated_v2ds,
                                                           tabulated_v2dd)
        
        shorten_tabulated_site_index = np.mod(enlarged_atom_inx,
                                              len(input_image))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        # mulliken 1st nbr shell
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 1**0.5*np.sort(nbr_dis)[1]+0.01))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        idx_mulliken = (self.material_dict[idx_sym][b'IonizationPotential']
                        + self.material_dict[idx_sym][b'ElectronAffinity']) / 2.0
        
        nbr_mulliken = np.array([self.material_dict[nbr_sym][b'IonizationPotential']
                                 + self.material_dict[nbr_sym][b'ElectronAffinity']
                                 for nbr_sym in nbr_syms]) / 2.0
        
        tabulated_mulliken = (idx_mulliken - np.prod(nbr_mulliken)
                              **(1/len(nbr_mulliken)))
        
        ans = []
        
        for idx_model in range(0,10):
            model = Prediction(shorten_atom_fea,
                               shorten_nbr_fea,
                               shorten_nbr_fea_idx,
                               idx_model=idx_model,
                               tabulated_filling_inf=tabulated_filling_inf,
                               tabulated_d_cen_inf=tabulated_d_cen_inf,
                               tabulated_padding_fillter=tabulated_padding_fillter,
                               tabulated_full_width_inf=tabulated_full_width_inf,
                               tabulated_mulliken=tabulated_mulliken,
                               tabulated_site_index=shorten_tabulated_site_index,
                               tabulated_v2dd=tabulated_v2dd_sorted,
                               tabulated_v2ds=tabulated_v2ds_sorted)
            
            ans += [model.predict_properties(return_all_parm=return_all_parm)]
        
        if return_all_parm == False:
            ans = np.stack(ans)[:,:,0]
            mean_center = np.mean(ans[:, 1])   # d-band center
            std_center = np.std(ans[:, 1])
            print(f"band center of the atom (index {atom_inx}) of {system_name}: {mean_center:.2f} ± {std_center:.2f} eV")
            return ans[:,1]
        else:
            nbr_rad = np.pad(nbr_rad,
                             [0, 86-len(nbr_rad)],
                             mode='constant',
                             constant_values=0)
            return (ans,
                    np.concatenate(([idx_rad],
                                    nbr_rad,
                                    tabulated_d_ij_sorted,
                                    [tabulated_d_cen_inf],
                                    [tabulated_full_width_inf],
                                    [tabulated_mulliken],
                                    tabulated_padding_fillter[atom_inx])))
    
    def image2band_filling(self,
                           input_image,
                           atom_inx,
                           return_all_parm=False):
        
        assert atom_inx + 1 <= len(input_image)
        
        enlarged_image = input_image.copy()
        enlarged_atom_inx = len(enlarged_image)*60 + atom_inx
        enlarged_image = enlarged_image.repeat((11,11,1))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        tabulated_filling_inf = self.material_dict[idx_sym]['bulk_filling']
        tabulated_d_cen_inf = self.material_dict[idx_sym]['d_cen']
        tabulated_full_width_inf = self.material_dict[idx_sym]['full_width']
        
        idx_rad = self.material_dict[idx_sym]['rd']
        nbr_rad = np.array([self.material_dict[nbr_sym]['rd'] for nbr_sym in nbr_syms])
        
        vds = idx_rad**1.5 / nbr_dis**3.5
        vdd = idx_rad**1.5 * nbr_rad**1.5 / nbr_dis**5.0
        
        v2ds = 9.9856 * vds**2.0 * 7.62**2
        v2dd = 415.565 * vdd**2.0 * 7.62**2
        
        assert len(v2ds) <= 86
        assert len(v2dd) <= 86
        assert len(nbr_dis) <= 86
        assert len(nbr_lists) <= 86
        
        tabulated_v2ds = np.pad(v2ds, [0, 86-len(v2ds)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_v2dd = np.pad(v2dd, [0, 86-len(v2dd)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_d_ij = np.pad(nbr_dis, [0, 86-len(nbr_dis)],
                                mode='constant',
                                constant_values=1E6)
        
        tabulated_nbr_idx = np.pad(nbr_lists, [0, 86-len(nbr_lists)],
                                   mode='constant',
                                   constant_values=1E6)
        
        (shorten_idx_syms,
         shorten_nbr_syms,
         shorten_atom_fea,
         shorten_nbr_fea,
         shorten_nbr_fea_idx,
         tabulated_d_ij_sorted,
         tabulated_nbr_index_sorted,
         tabulated_v2dd_sorted,
         tabulated_v2ds_sorted,
         tabulated_padding_fillter) = self.descriptor.feas(input_image,
                                                           enlarged_image,
                                                           tabulated_nbr_idx,
                                                           tabulated_d_ij,
                                                           enlarged_atom_inx,
                                                           tabulated_v2ds,
                                                           tabulated_v2dd)
        
        shorten_tabulated_site_index = np.mod(enlarged_atom_inx,
                                              len(input_image))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        # mulliken 1st nbr shell
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 1**0.5*np.sort(nbr_dis)[1]+0.01))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        idx_mulliken = (self.material_dict[idx_sym][b'IonizationPotential']
                        + self.material_dict[idx_sym][b'ElectronAffinity']) / 2.0
        
        nbr_mulliken = np.array([self.material_dict[nbr_sym][b'IonizationPotential']
                                 + self.material_dict[nbr_sym][b'ElectronAffinity']
                                 for nbr_sym in nbr_syms]) / 2.0
        
        tabulated_mulliken = (idx_mulliken - np.prod(nbr_mulliken)
                              **(1/len(nbr_mulliken)))
        
        ans = []
        
        for idx_model in range(0,10):
            model = Prediction(shorten_atom_fea,
                               shorten_nbr_fea,
                               shorten_nbr_fea_idx,
                               idx_model=idx_model,
                               tabulated_filling_inf=tabulated_filling_inf,
                               tabulated_d_cen_inf=tabulated_d_cen_inf,
                               tabulated_padding_fillter=tabulated_padding_fillter,
                               tabulated_full_width_inf=tabulated_full_width_inf,
                               tabulated_mulliken=tabulated_mulliken,
                               tabulated_site_index=shorten_tabulated_site_index,
                               tabulated_v2dd=tabulated_v2dd_sorted,
                               tabulated_v2ds=tabulated_v2ds_sorted)
            
            ans += [model.predict_properties(return_all_parm=return_all_parm)]
        
        if return_all_parm == False:
            ans = np.stack(ans)[:,:,0]
            mean_filling = np.mean(ans[:, 0])  # d-band filling
            std_filling = np.std(ans[:, 0])
            print(f"band filling: {mean_filling:.6f} ± {std_filling:.6f}")
            return ans[:,0]
        else:
            nbr_rad = np.pad(nbr_rad,
                             [0, 86-len(nbr_rad)],
                             mode='constant',
                             constant_values=0)
            return (ans,
                    np.concatenate(([idx_rad],
                                    nbr_rad,
                                    tabulated_d_ij_sorted,
                                    [tabulated_d_cen_inf],
                                    [tabulated_full_width_inf],
                                    [tabulated_mulliken],
                                    tabulated_padding_fillter[atom_inx])))

    def image2band_center(self,
                          image,
                          atom_inx,
                          return_all_parm=False):
        
        image = _copy_without_adsorbates(image)
        
        if atom_inx + 1 > len(image):
            raise IndexError(f"atom_inx={atom_inx} is outside the clean image with {len(image)} atoms.")
        
        enlarged_image = image.copy()
        enlarged_atom_inx = len(enlarged_image)*60 + atom_inx
        enlarged_image = enlarged_image.repeat((11,11,1))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        tabulated_filling_inf = self.material_dict[idx_sym]['bulk_filling']
        tabulated_d_cen_inf = self.material_dict[idx_sym]['d_cen']
        tabulated_full_width_inf = self.material_dict[idx_sym]['full_width']
        
        idx_rad = self.material_dict[idx_sym]['rd']
        
        nbr_rad = np.array([self.material_dict[nbr_sym]['rd'] for nbr_sym in nbr_syms])
        
        vds = idx_rad**1.5 / nbr_dis**3.5
        vdd = idx_rad**1.5 * nbr_rad**1.5 / nbr_dis**5.0
        
        v2ds = 9.9856 * vds**2.0 * 7.62**2
        v2dd = 415.565 * vdd**2.0 * 7.62**2
        
        assert len(v2ds) <= 86
        assert len(v2dd) <= 86
        assert len(nbr_dis) <= 86
        assert len(nbr_lists) <= 86
        
        tabulated_v2ds = np.pad(v2ds, [0, 86-len(v2ds)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_v2dd = np.pad(v2dd, [0, 86-len(v2dd)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_d_ij = np.pad(nbr_dis, [0, 86-len(nbr_dis)],
                                mode='constant',
                                constant_values=1E6)
        
        tabulated_nbr_idx = np.pad(nbr_lists, [0, 86-len(nbr_lists)],
                                   mode='constant',
                                   constant_values=1E6)
        
        (shorten_idx_syms,
         shorten_nbr_syms,
         shorten_atom_fea,
         shorten_nbr_fea,
         shorten_nbr_fea_idx,
         tabulated_d_ij_sorted,
         tabulated_nbr_index_sorted,
         tabulated_v2dd_sorted,
         tabulated_v2ds_sorted,
         tabulated_padding_fillter) = self.descriptor.feas(image,
                                                           enlarged_image,
                                                           tabulated_nbr_idx,
                                                           tabulated_d_ij,
                                                           enlarged_atom_inx,
                                                           tabulated_v2ds,
                                                           tabulated_v2dd)
        
        shorten_tabulated_site_index = np.mod(enlarged_atom_inx,
                                              len(image))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        # mulliken 1st nbr shell
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 1**0.5*np.sort(nbr_dis)[1]+0.01))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        idx_mulliken = (self.material_dict[idx_sym][b'IonizationPotential']
                        + self.material_dict[idx_sym][b'ElectronAffinity']) / 2.0
        
        nbr_mulliken = np.array([self.material_dict[nbr_sym][b'IonizationPotential']
                                 + self.material_dict[nbr_sym][b'ElectronAffinity']
                                 for nbr_sym in nbr_syms]) / 2.0
        
        tabulated_mulliken = (idx_mulliken - np.prod(nbr_mulliken)
                              **(1/len(nbr_mulliken)))
        
        ans = []
        
        for idx_model in range(0,10):
            model = Prediction(shorten_atom_fea,
                               shorten_nbr_fea,
                               shorten_nbr_fea_idx,
                               idx_model=idx_model,
                               tabulated_filling_inf=tabulated_filling_inf,
                               tabulated_d_cen_inf=tabulated_d_cen_inf,
                               tabulated_padding_fillter=tabulated_padding_fillter,
                               tabulated_full_width_inf=tabulated_full_width_inf,
                               tabulated_mulliken=tabulated_mulliken,
                               tabulated_site_index=shorten_tabulated_site_index,
                               tabulated_v2dd=tabulated_v2dd_sorted,
                               tabulated_v2ds=tabulated_v2ds_sorted)
            
            ans += [model.predict_properties(return_all_parm=return_all_parm)]
        
        if return_all_parm == False:
            ans = np.stack(ans)[:,:,0]
            mean_center = np.mean(ans[:, 1])   # d-band center
            std_center = np.std(ans[:, 1])
            #print(f"band center: {mean_center:.6f} ± {std_center:.6f} eV")
            return ans[:,1]
        else:
            nbr_rad = np.pad(nbr_rad,
                             [0, 86-len(nbr_rad)],
                             mode='constant',
                             constant_values=0)
            return (ans,
                    np.concatenate(([idx_rad],
                                    nbr_rad,
                                    tabulated_d_ij_sorted,
                                    [tabulated_d_cen_inf],
                                    [tabulated_full_width_inf],
                                    [tabulated_mulliken],
                                    tabulated_padding_fillter[atom_inx])))
    
    def image2band_full_rectangular_width(self,
                                          image,
                                          atom_inx,
                                          return_all_parm=False):
        
        assert atom_inx + 1 <= len(image)
        
        enlarged_image = image.copy()
        enlarged_atom_inx = len(enlarged_image)*60 + atom_inx
        enlarged_image = enlarged_image.repeat((11,11,1))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        tabulated_filling_inf = self.material_dict[idx_sym]['bulk_filling']
        tabulated_d_cen_inf = self.material_dict[idx_sym]['d_cen']
        tabulated_full_width_inf = self.material_dict[idx_sym]['full_width']
        
        idx_rad = self.material_dict[idx_sym]['rd']
        nbr_rad = np.array([self.material_dict[nbr_sym]['rd'] for nbr_sym in nbr_syms])
        
        vds = idx_rad**1.5 / nbr_dis**3.5
        vdd = idx_rad**1.5 * nbr_rad**1.5 / nbr_dis**5.0
        
        v2ds = 9.9856 * vds**2.0 * 7.62**2
        v2dd = 415.565 * vdd**2.0 * 7.62**2
        
        assert len(v2ds) <= 86
        assert len(v2dd) <= 86
        assert len(nbr_dis) <= 86
        assert len(nbr_lists) <= 86
        
        tabulated_v2ds = np.pad(v2ds, [0, 86-len(v2ds)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_v2dd = np.pad(v2dd, [0, 86-len(v2dd)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_d_ij = np.pad(nbr_dis, [0, 86-len(nbr_dis)],
                                mode='constant',
                                constant_values=1E6)
        
        tabulated_nbr_idx = np.pad(nbr_lists, [0, 86-len(nbr_lists)],
                                   mode='constant',
                                   constant_values=1E6)
        
        (shorten_idx_syms,
         shorten_nbr_syms,
         shorten_atom_fea,
         shorten_nbr_fea,
         shorten_nbr_fea_idx,
         tabulated_d_ij_sorted,
         tabulated_nbr_index_sorted,
         tabulated_v2dd_sorted,
         tabulated_v2ds_sorted,
         tabulated_padding_fillter) = self.descriptor.feas(image,
                                                           enlarged_image,
                                                           tabulated_nbr_idx,
                                                           tabulated_d_ij,
                                                           enlarged_atom_inx,
                                                           tabulated_v2ds,
                                                           tabulated_v2dd)
        
        shorten_tabulated_site_index = np.mod(enlarged_atom_inx,
                                              len(image))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        # mulliken 1st nbr shell
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 1**0.5*np.sort(nbr_dis)[1]+0.01))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        idx_mulliken = (self.material_dict[idx_sym][b'IonizationPotential']
                        + self.material_dict[idx_sym][b'ElectronAffinity']) / 2.0
        
        nbr_mulliken = np.array([self.material_dict[nbr_sym][b'IonizationPotential']
                                 + self.material_dict[nbr_sym][b'ElectronAffinity']
                                 for nbr_sym in nbr_syms]) / 2.0
        
        tabulated_mulliken = (idx_mulliken - np.prod(nbr_mulliken)
                              **(1/len(nbr_mulliken)))
        
        ans = []
        
        for idx_model in range(0,10):
            model = Prediction(shorten_atom_fea,
                               shorten_nbr_fea,
                               shorten_nbr_fea_idx,
                               idx_model=idx_model,
                               tabulated_filling_inf=tabulated_filling_inf,
                               tabulated_d_cen_inf=tabulated_d_cen_inf,
                               tabulated_padding_fillter=tabulated_padding_fillter,
                               tabulated_full_width_inf=tabulated_full_width_inf,
                               tabulated_mulliken=tabulated_mulliken,
                               tabulated_site_index=shorten_tabulated_site_index,
                               tabulated_v2dd=tabulated_v2dd_sorted,
                               tabulated_v2ds=tabulated_v2ds_sorted)
            
            ans += [model.predict_properties(return_all_parm=return_all_parm)]
        
        if return_all_parm == False:
            ans = np.stack(ans)[:,:,0]
            mean_width = np.mean(ans[:, 2])    # d-band full width
            std_width = np.std(ans[:, 2])
            #print(f"band full rectangular width: {mean_width:.6f} ± {std_width:.6f} eV")
            return ans[:,2]
        else:
            nbr_rad = np.pad(nbr_rad,
                             [0, 86-len(nbr_rad)],
                             mode='constant',
                             constant_values=0)
            return (ans,
                    np.concatenate(([idx_rad],
                                    nbr_rad,
                                    tabulated_d_ij_sorted,
                                    [tabulated_d_cen_inf],
                                    [tabulated_full_width_inf],
                                    [tabulated_mulliken],
                                    tabulated_padding_fillter[atom_inx])))
    
    def image2dcen(self,
                   image,
                   atom_inx,
                   return_all_parm=False):
        
        image = _copy_without_adsorbates(image)
        
        if atom_inx + 1 > len(image):
            raise IndexError(f"atom_inx={atom_inx} is outside the clean image with {len(image)} atoms.")
        
        enlarged_image = image.copy()
        enlarged_atom_inx = len(enlarged_image)*60 + atom_inx
        enlarged_image = enlarged_image.repeat((11,11,1))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        tabulated_filling_inf = self.material_dict[idx_sym]['bulk_filling']
        tabulated_d_cen_inf = self.material_dict[idx_sym]['d_cen']
        tabulated_full_width_inf = self.material_dict[idx_sym]['full_width']
        
        idx_rad = self.material_dict[idx_sym]['rd']
        nbr_rad = np.array([self.material_dict[nbr_sym]['rd'] for nbr_sym in nbr_syms])
        
        vds = idx_rad**1.5 / nbr_dis**3.5
        vdd = idx_rad**1.5 * nbr_rad**1.5 / nbr_dis**5.0
        
        v2ds = 9.9856 * vds**2.0 * 7.62**2
        v2dd = 415.565 * vdd**2.0 * 7.62**2
        
        assert len(v2ds) <= 86
        assert len(v2dd) <= 86
        assert len(nbr_dis) <= 86
        assert len(nbr_lists) <= 86
        
        tabulated_v2ds = np.pad(v2ds, [0, 86-len(v2ds)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_v2dd = np.pad(v2dd, [0, 86-len(v2dd)],
                                mode='constant',
                                constant_values=0)
        
        tabulated_d_ij = np.pad(nbr_dis, [0, 86-len(nbr_dis)],
                                mode='constant',
                                constant_values=1E6)
        
        tabulated_nbr_idx = np.pad(nbr_lists, [0, 86-len(nbr_lists)],
                                   mode='constant',
                                   constant_values=1E6)
        
        (shorten_idx_syms,
         shorten_nbr_syms,
         shorten_atom_fea,
         shorten_nbr_fea,
         shorten_nbr_fea_idx,
         tabulated_d_ij_sorted,
         tabulated_nbr_index_sorted,
         tabulated_v2dd_sorted,
         tabulated_v2ds_sorted,
         tabulated_padding_fillter) = self.descriptor.feas(image,
                                                           enlarged_image,
                                                           tabulated_nbr_idx,
                                                           tabulated_d_ij,
                                                           enlarged_atom_inx,
                                                           tabulated_v2ds,
                                                           tabulated_v2dd)
        
        shorten_tabulated_site_index = np.mod(enlarged_atom_inx,
                                              len(image))
        
        nbr_dis = enlarged_image.get_distances(enlarged_atom_inx,
                                               list(range(len(enlarged_image))))
        
        # mulliken 1st nbr shell
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 1**0.5*np.sort(nbr_dis)[1]+0.01))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_atom_inx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        idx_mulliken = (self.material_dict[idx_sym][b'IonizationPotential']
                        + self.material_dict[idx_sym][b'ElectronAffinity']) / 2.0
        
        nbr_mulliken = np.array([self.material_dict[nbr_sym][b'IonizationPotential']
                                 + self.material_dict[nbr_sym][b'ElectronAffinity']
                                 for nbr_sym in nbr_syms]) / 2.0
        
        tabulated_mulliken = (idx_mulliken - np.prod(nbr_mulliken)
                              **(1/len(nbr_mulliken)))
        
        ans = []
        
        for idx_model in range(0,10):
            model = Prediction(shorten_atom_fea,
                               shorten_nbr_fea,
                               shorten_nbr_fea_idx,
                               idx_model=idx_model,
                               tabulated_filling_inf=tabulated_filling_inf,
                               tabulated_d_cen_inf=tabulated_d_cen_inf,
                               tabulated_padding_fillter=tabulated_padding_fillter,
                               tabulated_full_width_inf=tabulated_full_width_inf,
                               tabulated_mulliken=tabulated_mulliken,
                               tabulated_site_index=shorten_tabulated_site_index,
                               tabulated_v2dd=tabulated_v2dd_sorted,
                               tabulated_v2ds=tabulated_v2ds_sorted)
            
            ans += [model.predict_d_cen(return_all_parm=return_all_parm)]
        
        if return_all_parm == False:
            mean = np.mean(ans)
            std = np.std(ans)
            print(f"band center: {mean:.6f} ± {std:.6f}")
            return np.stack(ans)
        else:
            nbr_rad = np.pad(nbr_rad,
                             [0, 86-len(nbr_rad)],
                             mode='constant',
                             constant_values=0)
            return (ans,
                    np.concatenate(([idx_rad],
                                    nbr_rad,
                                    tabulated_d_ij_sorted,
                                    [tabulated_d_cen_inf],
                                    [tabulated_full_width_inf],
                                    [tabulated_mulliken],
                                    tabulated_padding_fillter[atom_inx])))
    
    def gen_shap(self,
                 ref_image,
                 ref_atom_inx,
                 target_image,
                 target_atom_inx):
        
        (predicted_parameter_reference,
         tabulated_parameter_reference) = self.image2dcen(ref_image,
                                                          ref_atom_inx,
                                                          return_all_parm=True)
        
        (predicted_parameter_target,
         tabulated_parameter_target) = self.image2dcen(target_image,
                                                       target_atom_inx,
                                                       return_all_parm=True)
        
        shap_ligand = []
        shap_strain = []
        shap_relax = []
        shap_resonance = []
        shap_elect_transf = []
        predicted_d_cen_reference = []
        predicted_d_cen_target = []
        
        for i in range(0,10):
            
            inp_shap_reference = np.atleast_2d(
                np.hstack((tabulated_parameter_reference,
                           predicted_parameter_reference[i][2],
                           predicted_parameter_reference[i][3])))
            
            explainer = shap.Explainer(self.tinnet_d_center,
                                       inp_shap_reference)
            
            inp_shap_target = np.atleast_2d(
                np.hstack((tabulated_parameter_target,
                           predicted_parameter_target[i][2],
                           predicted_parameter_target[i][3])))
            
            shap_values = explainer(inp_shap_target).values
            
            #shap_idx_rad = shap_values[:,0]
            shap_nbr_rad = np.sum(shap_values[:,1:87]) # ligand
            shap_tabulated_d_ij_sorted = np.sum(shap_values[:,87:173]) # strain
            #shap_tabulated_d_cen_inf = shap_values[:,173]
            #shap_tabulated_full_width_inf = shap_values[:,174]
            shap_tabulated_mulliken = shap_values[:,175] # elect. transf
            #shap_tabulated_padding_fillter = np.sum(shap_values[:,176:262])
            shap_zeta = np.sum(shap_values[:,262:348]) # relax
            shap_alpha = shap_values[:,349] # resonance
            shap_beta = shap_values[:,348] # elect. transf
            
            shap_ligand += [shap_nbr_rad]
            shap_strain += [shap_tabulated_d_ij_sorted]
            shap_relax += [shap_zeta]
            shap_resonance += [shap_alpha]
            shap_elect_transf += [shap_beta + shap_tabulated_mulliken]
            
            predicted_d_cen_reference += [predicted_parameter_reference[i][0]]
            predicted_d_cen_target += [predicted_parameter_target[i][0]]
        
        return np.vstack((np.array(predicted_d_cen_reference).flatten(),
                          np.array(predicted_d_cen_target).flatten(),
                          np.array(shap_resonance).flatten(),
                          np.array(shap_strain),
                          np.array(shap_relax),
                          np.array(shap_ligand),
                          np.array(shap_elect_transf).flatten()))
    
    def tinnet_d_center(self,
                        inp_shap):
        idx_rad = inp_shap[:,0]
        nbr_rad = inp_shap[:,1:87]
        tabulated_d_ij_sorted = inp_shap[:,87:173]
        tabulated_d_cen_inf = inp_shap[:,173]
        tabulated_full_width_inf = inp_shap[:,174]
        tabulated_mulliken = inp_shap[:,175]
        tabulated_padding_fillter = inp_shap[:,176:262]
        zeta = inp_shap[:,262:348]
        crys_fea = inp_shap[:,348:350]
        
        vds = idx_rad[:,None]**1.5 / tabulated_d_ij_sorted**3.5
        vdd = idx_rad[:,None]**1.5 * nbr_rad**1.5 / tabulated_d_ij_sorted**5.0
        
        v2ds = 9.9856 * vds**2.0 * 7.62**2 * tabulated_padding_fillter
        v2dd = 415.565 * vdd**2.0 * 7.62**2 * tabulated_padding_fillter
        
        m2 = np.sum(v2ds / zeta**(7.0)
                    + v2dd / zeta**(10.0), axis=1)
        
        d_cen_tinnet = (crys_fea[:,1]
                        * m2**0.5
                        * (tabulated_d_cen_inf / tabulated_full_width_inf
                           - crys_fea[:,0] * tabulated_mulliken))
        
        return np.atleast_1d(d_cen_tinnet)
    
    def explain_shap(self,
                     ref_image=None,
                     ref_atom_inx=None,
                     ref_name='Reference',
                     plot_name='shap',
                     save_fig='png'):
        
        target_image = self.image
        target_atom_inx = self.atom_inx
        target_name = self.name
        
        rcParams['ps.useafm'] = True
        plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
        rcParams['pdf.fonttype'] = 42
        rcParams['errorbar.capsize'] = 4.0
        mpl.rcParams['ytick.major.width'] = 0.5
        mpl.rcParams['ytick.minor.width'] = 0.5
        mpl.rcParams['xtick.major.width'] = 0.5
        mpl.rcParams['xtick.minor.width'] = 0.5
        matplotlib.rc('xtick.major', size=4)
        matplotlib.rc('xtick.minor', size=2)
        matplotlib.rc('ytick.major', size=4)
        matplotlib.rc('ytick.minor', size=2)
        matplotlib.rc('lines', linewidth = 0.5)
        matplotlib.rc('lines', markeredgewidth=0.5)
        matplotlib.rc('font', size=7)
        plt.rcParams['axes.linewidth'] = 0.5
        
        fig, ax = plt.subplots()
        fig.set_size_inches(3.375*2.0, 3.375)
        
        shap = self.gen_shap(ref_image,
                             ref_atom_inx,
                             target_image,
                             target_atom_inx)
        
        shap = np.average(shap, axis=1)
        
        dx5 = shap[2]
        dx4 = shap[3]
        dx3 = shap[4]
        dx2 = shap[5]
        dx1 = shap[6]
        
        x5 = shap[0]
        x4 = x5 + dx5
        x3 = x4 + dx4
        x2 = x3 + dx3
        x1 = x2 + dx2
        
        c5 = (shap[2] < 0) * 'red' + (shap[2] >= 0) * 'blue'
        c4 = (shap[3] < 0) * 'red' + (shap[3] >= 0) * 'blue'
        c3 = (shap[4] < 0) * 'red' + (shap[4] >= 0) * 'blue'
        c2 = (shap[5] < 0) * 'red' + (shap[5] >= 0) * 'blue'
        c1 = (shap[6] < 0) * 'red' + (shap[6] >= 0) * 'blue'
        
        ax.arrow(x=x5, y=5, dx=dx5, dy=0, color=c5, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx5),
                 length_includes_head=True)
        ax.arrow(x=x4, y=4, dx=dx4, dy=0, color=c4, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx4),
                 length_includes_head=True)
        ax.arrow(x=x3, y=3, dx=dx3, dy=0, color=c3, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx3),
                 length_includes_head=True)
        ax.arrow(x=x2, y=2, dx=dx2, dy=0, color=c2, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx2),
                 length_includes_head=True)
        ax.arrow(x=x1, y=1, dx=dx1, dy=0, color=c1, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx1),
                 length_includes_head=True)
        
        ax.set_ylim([0.5, 5.5])
        
        labels = [r'$(\alpha, \xi)$',
                  r'$d_{ij}$',
                  r'$\zeta$',
                  r'$(\lambda, r_{dj})$',
                  r'$(\beta, \Delta\chi)$']
        
        plt.yticks([5,4,3,2,1], labels)
        
        ax.annotate('Resonance', xy=(-0.12, 5.0),
                    xycoords=('axes fraction', 'data'),
                    ha='center', va='center')
        ax.annotate('Ligand', xy=(-0.12, 2.0),
                    xycoords=('axes fraction', 'data'),
                    ha='center', va='center')
        ax.annotate('Charge\ntransfer', xy=(-0.12, 1.0),
                    xycoords=('axes fraction', 'data'),
                    ha='center', va='center')
        
        ax.annotate('Strain',
                    xy=(-0.06, 3.5), xycoords=('axes fraction', 'data'),
                    xytext=(-0.12, 3.5), textcoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=1.0'),
                    ha='center', va='center')
        
        sym_5 = (shap[2] < 0) * '-' + (shap[2] >= 0) * '+'
        sym_4 = (shap[3] < 0) * '-' + (shap[3] >= 0) * '+'
        sym_3 = (shap[4] < 0) * '-' + (shap[4] >= 0) * '+'
        sym_2 = (shap[5] < 0) * '-' + (shap[5] >= 0) * '+'
        sym_1 = (shap[6] < 0) * '-' + (shap[6] >= 0) * '+'
        
        ax.annotate(sym_5 + '{:.2f}'.format(round(np.abs(shap[2]), 4)),
                    xy=(-0.12, 4.70), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c5)
        ax.annotate(sym_4 + '{:.2f}'.format(round(np.abs(shap[3]), 4)),
                    xy=(-0.12, 3.80), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c4)
        ax.annotate(sym_3 + '{:.2f}'.format(round(np.abs(shap[4]), 4)),
                    xy=(-0.12, 3.20), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c3)
        ax.annotate(sym_2 + '{:.2f}'.format(round(np.abs(shap[5]), 4)),
                    xy=(-0.12, 1.70), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c2)
        ax.annotate(sym_1 + '{:.2f}'.format(round(np.abs(shap[6]), 4)),
                    xy=(-0.12, 0.60), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c1)
        
        ax.set_xlabel(r'$d\rm{-band\ center}$ (eV)')
        ax.spines[['left', 'right', 'top']].set_visible(False)
        
        ax.tick_params('y', length=0, width=0, which='major')
        
        ax.plot([x5, x5],[7, 0.5],'--', color='gray', linewidth=1)
        
        ax.plot([x4, x4], [5 + 1.0 / 3.0, 4 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        ax.plot([x3, x3], [4 + 1.0 / 3.0, 3 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        ax.plot([x2, x2], [3 + 1.0 / 3.0, 2 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        ax.plot([x1, x1], [2 + 1.0 / 3.0, 1 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        
        ax.plot([x1 + dx1, x1 + dx1],[7, 0.5],'--', color='orange',
                linewidth=1)
        
        ax.annotate(ref_name + '\n{:.2f}'.format(round(x5, 4)),
                    xy=(x5, 1.15), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='gray')
        ax.annotate(target_name + '\n{:.2f}'.format(round(x1 + dx1, 4)),
                    xy=(x1 + dx1, 1.05), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='orange')
        
        fig.tight_layout()
        
        if save_fig == 'png':
            fig.savefig(plot_name + '.png', bbox_inches='tight', dpi=600)
        elif save_fig == 'pdf':
            fig.savefig(plot_name + '.pdf', bbox_inches='tight')


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
        
        cuda = torch.cuda.is_available()
        
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
        self.crystal_atom_idx = torch.from_numpy(np.arange(atom_fea.shape[0]))
    
    def predict_d_cen(self,
                      return_all_parm=False,
                      **kwargs):
        
        if self.cuda:
            best_checkpoint = torch.load('./data/pretrained/band_center/model_' + str(self.idx_model) + '.pth.tar')
        else:
            best_checkpoint = torch.load('./data/pretrained/band_center/model_' + str(self.idx_model) + '.pth.tar', map_location=torch.device('cpu'))
        
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        # switch to evaluate mode
        self.model.eval()
        
        with torch.no_grad():
            if self.cuda:
                input_var = (Variable(self.atom_fea.cuda(non_blocking=True)),
                             Variable(self.nbr_fea.cuda(non_blocking=True)),
                             self.nbr_fea_idx.cuda(non_blocking=True),
                             self.tabulated_padding_fillter.cuda(non_blocking=True),
                             self.crystal_atom_idx.cuda(non_blocking=True),
                             self.tabulated_site_index.cuda(non_blocking=True))
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
            best_checkpoint = torch.load('./data/pretrained/band_center/model_' + str(self.idx_model) + '.pth.tar')
        else:
            best_checkpoint = torch.load('./data/pretrained/band_center/model_' + str(self.idx_model) + '.pth.tar', map_location=torch.device('cpu'))
        
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        # switch to evaluate mode
        self.model.eval()
        
        with torch.no_grad():
            if self.cuda:
                input_var = (Variable(self.atom_fea.cuda(non_blocking=True)),
                             Variable(self.nbr_fea.cuda(non_blocking=True)),
                             self.nbr_fea_idx.cuda(non_blocking=True),
                             self.tabulated_padding_fillter.cuda(non_blocking=True),
                             self.crystal_atom_idx.cuda(non_blocking=True),
                             self.tabulated_site_index.cuda(non_blocking=True))
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
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, tabulated_padding_fillter, crystal_atom_idx, atom_inx):
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
            self.dict_atom_fea = self.dict_atom_fea_default()
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
    
    def dict_atom_fea_default(self):
        
        # Default superstructure of atom features
        atom_fea_dict = {
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0],
            3: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0],
            4: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0],
            10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            11: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            12: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            14: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            17: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            19: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            20: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            21: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            22: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            23: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            24: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            25: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            26: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            27: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            28: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            29: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            30: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            31: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            32: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            33: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            34: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            36: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            37: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            38: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            39: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            40: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            41: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            42: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            43: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            44: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            45: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            46: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            47: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            48: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            49: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            50: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            51: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            52: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            53: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            54: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            55: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            56: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            57: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            58: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            59: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            60: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            61: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            62: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            63: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            64: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            65: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            66: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            67: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            68: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            69: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            70: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            71: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            72: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            73: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            74: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            75: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            76: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            77: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            78: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            79: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            80: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            81: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            82: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            83: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            84: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            85: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            86: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            87: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            88: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            89: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            90: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            91: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            92: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            93: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            94: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            95: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            96: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            97: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            98: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            99: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            100: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        return atom_fea_dict
    
    def material_dict():
        # Element name
        # d-band center, eV
        # d-band full width, eV
        # rd, A
        # bulk_filling, -
        
        reference_data = {'Ag':{'d_cen':-4.1421, 'full_width': 4.5040, 'rd': 0.6606606606606606, 'bulk_filling': 9.755790629075912213e-01},
                          'Au':{'d_cen':-3.6577, 'full_width': 5.8312, 'rd': 0.7607607607607607, 'bulk_filling': 9.638752405788268973e-01},
                          #'Cd':{'d_cen':-8.6701, 'full_width': 3.4283, 'bulk_filling': 9.931763504350845650e-01},
                          'Co':{'d_cen':-1.5905, 'full_width': 6.6969, 'rd': 0.5605605605605606, 'bulk_filling': 7.711781551994808526e-01},
                          'Cr':{'d_cen':-0.0876, 'full_width': 7.2096, 'rd': 0.6306306306306306, 'bulk_filling': 5.026890307897071697e-01},
                          'Cu':{'d_cen':-2.6521, 'full_width': 4.2446, 'rd': 0.4904904904904905, 'bulk_filling': 9.687762969001947333e-01},
                          'Fe':{'d_cen':-0.9278, 'full_width': 7.1120, 'rd': 0.6206206206206206, 'bulk_filling': 6.842646244058228078e-01},
                          #'Hf':{'d_cen': 2.0669, 'full_width':10.4322, 'bulk_filling': 2.798683666831001671e-01},
                          'Ir':{'d_cen':-2.6636, 'full_width': 9.5022, 'rd': 0.8208208208208209, 'bulk_filling': 7.713775689575533834e-01},
                          #'La':{'d_cen': 2.0866, 'full_width': 8.0907, 'bulk_filling': 2.196169569958589252e-01},
                          'Mn':{'d_cen':-0.6036, 'full_width': 7.0846, 'rd': 0.5905905905905906, 'bulk_filling': 5.625623395638867930e-01},
                          'Mo':{'d_cen':-0.0110, 'full_width': 9.0573, 'rd': 0.8608608608608609, 'bulk_filling': 5.085650613858612168e-01},
                          'Nb':{'d_cen': 0.6878, 'full_width': 8.9762, 'rd': 0.9409409409409409, 'bulk_filling': 4.053595224934098407e-01},
                          'Ni':{'d_cen':-1.6686, 'full_width': 5.3541, 'rd': 0.5205205205205206, 'bulk_filling': 8.642030365513106993e-01},
                          'Os':{'d_cen':-1.9693, 'full_width':10.6399, 'rd': 0.8508508508508509, 'bulk_filling': 6.868617591064947181e-01},
                          'Pd':{'d_cen':-2.0870, 'full_width': 5.6793, 'rd': 0.6706706706706707, 'bulk_filling': 9.147420170513048676e-01},
                          'Pt':{'d_cen':-2.6369, 'full_width': 7.6381, 'rd': 0.7907907907907907, 'bulk_filling': 8.750916806647935919e-01},
                          'Re':{'d_cen':-1.0633, 'full_width':10.9953, 'rd': 0.8808808808808809, 'bulk_filling': 5.905781396979362663e-01},
                          'Rh':{'d_cen':-2.0874, 'full_width': 7.3926, 'rd': 0.7207207207207207, 'bulk_filling': 7.951773878664225581e-01},
                          'Ru':{'d_cen':-1.6305, 'full_width': 8.0663, 'rd': 0.7507507507507507, 'bulk_filling': 7.060335740069819677e-01},
                          'Sc':{'d_cen': 1.7032, 'full_width': 5.9997, 'rd': 0.9409409409409409, 'bulk_filling': 1.936272929841076074e-01},
                          'Ta':{'d_cen': 1.1906, 'full_width':10.8913, 'rd': 1.021021021021021 , 'bulk_filling': 3.786712857939578680e-01},
                          'Ti':{'d_cen': 1.3243, 'full_width': 6.7421, 'rd': 0.7907907907907907, 'bulk_filling': 2.653109539307398901e-01},
                          'V': {'d_cen': 0.5548, 'full_width': 6.9774, 'rd': 0.6906906906906907, 'bulk_filling': 3.818390980189110273e-01},
                          'W': {'d_cen': 0.1732, 'full_width':10.7373, 'rd': 0.9409409409409409, 'bulk_filling': 4.850581815594974255e-01},
                          'Y': {'d_cen': 2.2707, 'full_width': 7.8737, 'rd': 1.2512512512512513, 'bulk_filling': 1.875853315789731690e-01},
                          #'Zn':{'d_cen':-7.3926, 'full_width': 2.4747, 'bulk_filling': 9.979212627042637340e-01},
                          'Zr':{'d_cen': 1.6485, 'full_width': 8.9050, 'rd': 1.0710710710710711, 'bulk_filling': 2.920224652312821134e-01}}
        
        with open('./data/MaterialDict.pkl', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        data = {k.decode('utf8'): v for k, v in data.items()}
        
        keys = reference_data.keys()
        
        properties = ['d_cen', 'full_width', 'rd', 'bulk_filling']
        
        for key in keys:
            for prop in properties:
                data[key][prop] = reference_data[key][prop]
        
        return data
