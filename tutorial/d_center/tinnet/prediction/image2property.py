#!/usr/bin/env python
# This script is adapted from Xie's and Ulissi's scripts.

import numpy as np
import shap

from ..feature.material_dict import Material_dict
from ..feature.voronoi import Voronoi
from ..prediction.prediction import Prediction


class Image2property:
    def __init__(self):
        descriptor = Voronoi(radius=8,
                             dmin=0,
                             step=0.2,
                             dict_atom_fea=None)

        data = Material_dict()
        
        bulk_filling = {'Nb': 4.053595224934098407e-01,
                        'Sc': 1.936272929841076074e-01,
                        'Co': 7.711781551994808526e-01,
                        'Y':  1.875853315789731690e-01,
                        'Cr': 5.026890307897071697e-01,
                        'Hf': 2.798683666831001671e-01,
                        'Cu': 9.687762969001947333e-01,
                        'Pd': 9.147420170513048676e-01,
                        'Pt': 8.750916806647935919e-01,
                        'Mo': 5.085650613858612168e-01,
                        'Rh': 7.951773878664225581e-01,
                        'Zn': 9.979212627042637340e-01,
                        'Ag': 9.755790629075912213e-01,
                        'W':  4.850581815594974255e-01,
                        'Ni': 8.642030365513106993e-01,
                        'Au': 9.638752405788268973e-01,
                        'Cd': 9.931763504350845650e-01,
                        'V':  3.818390980189110273e-01,
                        'Mn': 5.625623395638867930e-01,
                        'Re': 5.905781396979362663e-01,
                        'Ru': 7.060335740069819677e-01,
                        'Ta': 3.786712857939578680e-01,
                        'Ti': 2.653109539307398901e-01,
                        'Os': 6.868617591064947181e-01,
                        'La': 2.196169569958589252e-01,
                        'Fe': 6.842646244058228078e-01,
                        'Ir': 7.713775689575533834e-01,
                        'Zr': 2.920224652312821134e-01}
        
        self.descriptor = descriptor
        self.data = data
        self.bulk_filling = bulk_filling
    
    def image2dcen(self,
                   input_image,
                   site_idx,
                   return_all_parm=False):
        
        assert site_idx + 1 <= len(input_image)
        
        enlarged_image = input_image.copy()
        enlarged_site_idx = len(enlarged_image)*60 + site_idx
        enlarged_image = enlarged_image.repeat((11,11,1))
        
        nbr_dis = enlarged_image.get_distances(enlarged_site_idx,
                                               list(range(len(enlarged_image))))
        
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_site_idx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        tabulated_filling_inf = self.bulk_filling[idx_sym]
        tabulated_d_cen_inf = self.data[idx_sym]['d_cen']
        tabulated_full_width_inf = self.data[idx_sym]['full_width']
        
        idx_rad = self.data[idx_sym]['rd']
        nbr_rad = np.array([self.data[nbr_sym]['rd'] for nbr_sym in nbr_syms])
        
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
                                                           enlarged_site_idx,
                                                           tabulated_v2ds,
                                                           tabulated_v2dd)
        
        shorten_tabulated_site_index = np.mod(enlarged_site_idx,
                                              len(input_image))
        
        nbr_dis = enlarged_image.get_distances(enlarged_site_idx,
                                               list(range(len(enlarged_image))))
        
        # mulliken 1st nbr shell
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 1**0.5*np.sort(nbr_dis)[1]+0.01))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_site_idx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        idx_mulliken = (self.data[idx_sym][b'IonizationPotential']
                        + self.data[idx_sym][b'ElectronAffinity']) / 2.0
        
        nbr_mulliken = np.array([self.data[nbr_sym][b'IonizationPotential']
                                 + self.data[nbr_sym][b'ElectronAffinity']
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
                                    tabulated_padding_fillter[site_idx])))
    
    def image2properties(self,
                   input_image,
                   site_idx,
                   return_all_parm=False):
        
        assert site_idx + 1 <= len(input_image)
        
        enlarged_image = input_image.copy()
        enlarged_site_idx = len(enlarged_image)*60 + site_idx
        enlarged_image = enlarged_image.repeat((11,11,1))
        
        nbr_dis = enlarged_image.get_distances(enlarged_site_idx,
                                               list(range(len(enlarged_image))))
        
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 5.5))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_site_idx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        tabulated_filling_inf = self.bulk_filling[idx_sym]
        tabulated_d_cen_inf = self.data[idx_sym]['d_cen']
        tabulated_full_width_inf = self.data[idx_sym]['full_width']
        
        idx_rad = self.data[idx_sym]['rd']
        nbr_rad = np.array([self.data[nbr_sym]['rd'] for nbr_sym in nbr_syms])
        
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
                                                           enlarged_site_idx,
                                                           tabulated_v2ds,
                                                           tabulated_v2dd)
        
        shorten_tabulated_site_index = np.mod(enlarged_site_idx,
                                              len(input_image))
        
        nbr_dis = enlarged_image.get_distances(enlarged_site_idx,
                                               list(range(len(enlarged_image))))
        
        # mulliken 1st nbr shell
        nbr_lists = np.where((0.01 <= nbr_dis) * (nbr_dis <= 1**0.5*np.sort(nbr_dis)[1]+0.01))[0]
        
        nbr_dis = nbr_dis[nbr_lists]
        nbr_lists = nbr_lists[np.argsort(nbr_dis)]
        nbr_dis = np.sort(nbr_dis)
        
        idx_sym = enlarged_image[enlarged_site_idx].symbol
        nbr_syms = np.array([enlarged_image[nbr_list].symbol
                             for nbr_list in nbr_lists])
        
        idx_mulliken = (self.data[idx_sym][b'IonizationPotential']
                        + self.data[idx_sym][b'ElectronAffinity']) / 2.0
        
        nbr_mulliken = np.array([self.data[nbr_sym][b'IonizationPotential']
                                 + self.data[nbr_sym][b'ElectronAffinity']
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
            return np.stack(ans)[:,:,0]
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
                                    tabulated_padding_fillter[site_idx])))

    def calculate_shap(self,
                       reference_image,
                       reference_site_idx,
                       target_image,
                       target_site_idx):
        
        (predicted_parameter_reference,
         tabulated_parameter_reference) = self.image2dcen(reference_image,
                                                          reference_site_idx,
                                                          return_all_parm=True)
        
        (predicted_parameter_target,
         tabulated_parameter_target) = self.image2dcen(target_image,
                                                       target_site_idx,
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
    