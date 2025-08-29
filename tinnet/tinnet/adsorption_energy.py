#!/usr/bin/env python
# This script is adapted from Xie's and Ulissi's scripts.

import shap
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from ase import Atom
from pylab import *
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io.ase import AseAtomsAdaptor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from .band_center import BandCenter


class AdsorptionEnergy:
    def __init__(self,
                 image=None,
                 site_inx=None,
                 adsorbate=None,
                 name='Name'):
        self.image = image
        self.site_inx = site_inx
        self.adsorbate = adsorbate
        self.name = name
        self.atom_fea_dict = Features.dict_atom_fea_default(self)
        self.atom_prop_dict = Features.dict_atom_prop_default(self)
        self.descriptor = Features(max_num_nbr=12,
                                   radius=8,
                                   dmin=0,
                                   step=0.2,
                                   dict_atom_fea=None)
    
    def predict(self,
                image=None,
                site_inx=None,
                return_all_parm=False):
        
        if image == None:
            image = self.image
        if site_inx == None:
            site_inx = self.site_inx
        
        oh_indices = [i for i, atom in enumerate(image) if atom.symbol == 'O' or atom.symbol == 'H']
        for i in sorted(oh_indices, reverse=True):
            del image[i]
        
        # OH atop site
        if self.adsorbate == 'OH' and len(site_inx) == 1:
            self.phys_model = 'OH_atop'
            
            pos = image.get_positions()[site_inx[0]] + [0.0, 0.0, 2.00]
            image.append(Atom('O', position = pos))
            pos = image.get_positions()[site_inx[0]] + [0.8, 0.0, 2.41]
            image.append(Atom('H', position = pos))
            
            vad2 = np.array([self.atom_prop_dict[image.get_chemical_symbols()[i]]['vad2']
                             for i in site_inx], dtype=np.float32)
            
            results = [Regression(features=self.descriptor.feas(image),
                                  model_inx=model_inx,
                                  vad2=vad2,
                                  site_inx=site_inx,
                                  phys_model=self.phys_model).eval_model()
                       for model_inx in range(0,10)]
            
            model_ead, model_parm = zip(*results)
            model_ead = np.stack(list(model_ead)).flatten()
            model_parm = np.stack(list(model_parm))
            
            if return_all_parm == False:
                print(f"The adsorption energy of OH on the atop site (index {site_inx}) of {self.name}: {np.mean(model_ead):.2f} ± {np.std(model_ead):.2f} eV")
                return model_ead
            if return_all_parm == True:
                return model_parm
        
        # O atop site
        if self.adsorbate == 'O' and len(site_inx) == 1:
            self.phys_model = 'O_atop'
            
            model = BandCenter(image=image, atom_inx=site_inx[0])
            
            d_cen = model.image2band_center(image=image, atom_inx=site_inx[0])
            d_cen = np.average(d_cen)
            
            full_width = model.image2band_full_rectangular_width(image=image, atom_inx=site_inx[0])
            half_width = np.average(full_width) / np.sqrt(12) * 2.0
            
            pos = image.get_positions()[site_inx[0]] + [0.0, 0.0, 1.8]
            image.append(Atom('O', position = pos))
            
            
            vad2 = np.array([self.atom_prop_dict[image.get_chemical_symbols()[i]]['vad2']
                             for i in site_inx], dtype=np.float32)
            
            results = [Regression(features=self.descriptor.feas(image),
                                  model_inx=model_inx,
                                  vad2=vad2,
                                  site_inx=site_inx,
                                  d_cen=d_cen,
                                  half_width=half_width,
                                  phys_model=self.phys_model).eval_model()
                       for model_inx in range(0,10)]
            
            model_ead, model_parm = zip(*results)
            model_ead = np.stack(list(model_ead)).flatten()
            model_parm = np.stack(list(model_parm))
            
            if return_all_parm == False:
                print(f"The adsorption energy of O on the atop site (index {site_inx}) of {self.name}: {np.mean(model_ead):.2f} ± {np.std(model_ead):.2f} eV")
                return model_ead
            if return_all_parm == True:
                return model_parm
    
    def gen_shap(self,
                 ref_image,
                 ref_site_inx,
                 target_image,
                 target_site_inx):
        predicted_parameter_ref = self.predict(ref_image,
                                               ref_site_inx,
                                               return_all_parm=True)
        
        predicted_parameter_target = self.predict(target_image,
                                                  target_site_inx,
                                                  return_all_parm=True)
        parm_diff = np.average(predicted_parameter_target, axis=0) - np.average(predicted_parameter_ref, axis=0)
        
        if self.phys_model == 'OH_atop':
            shap_vad2 = []
            shap_d_cen = []
            shap_width = []
            shap_adse_1 = []
            shap_beta_1 = []
            shap_delta_1 = []
            shap_adse_2 = []
            shap_beta_2 = []
            shap_delta_2 = []
            shap_adse_3 = []
            shap_beta_3 = []
            shap_delta_3 = []
            
            predicted_d_cen_ref = []
            predicted_d_cen_target = []
            
            for i in range(0,10):
                
                inp_shap_ref = np.atleast_2d(predicted_parameter_ref[i][1:])
                
                explainer = shap.Explainer(self.tinnet_ead,
                                           inp_shap_ref)
                
                inp_shap_target = np.atleast_2d(predicted_parameter_target[i][1:])
                
                shap_values = explainer(inp_shap_target).values
                
                shap_vad2 += [shap_values[:,0]]
                shap_d_cen += [shap_values[:,1]]
                shap_width += [shap_values[:,2]]
                shap_adse_1 += [shap_values[:,3]]
                shap_beta_1 += [shap_values[:,4]]
                shap_delta_1 += [shap_values[:,5]]
                shap_adse_2 += [shap_values[:,6]]
                shap_beta_2 += [shap_values[:,7]]
                shap_delta_2 += [shap_values[:,8]]
                shap_adse_3 += [shap_values[:,9]]
                shap_beta_3 += [shap_values[:,10]]
                shap_delta_3 += [shap_values[:,11]]
                
                predicted_d_cen_ref += [predicted_parameter_ref[i][0]]
                predicted_d_cen_target += [predicted_parameter_target[i][0]]
            
            return np.vstack((np.array(predicted_d_cen_ref).flatten(),
                              np.array(predicted_d_cen_target).flatten(),
                              np.array(shap_vad2).flatten(),
                              np.array(shap_d_cen).flatten(),
                              np.array(shap_width).flatten(),
                              np.array(shap_adse_1).flatten(),
                              np.array(shap_beta_1).flatten(),
                              np.array(shap_delta_1).flatten(),
                              np.array(shap_adse_2).flatten(),
                              np.array(shap_beta_2).flatten(),
                              np.array(shap_delta_2).flatten(),
                              np.array(shap_adse_3).flatten(),
                              np.array(shap_beta_3).flatten(),
                              np.array(shap_delta_3).flatten())), parm_diff
        if self.phys_model == 'O_atop':
            shap_vad2 = []
            shap_d_cen = []
            shap_width = []
            shap_adse_2 = []
            shap_beta_2 = []
            shap_delta_2 = []
            shap_adse_3 = []
            shap_beta_3 = []
            shap_delta_3 = []
            
            predicted_d_cen_ref = []
            predicted_d_cen_target = []
            
            for i in range(0,10):
                
                inp_shap_ref = np.atleast_2d(predicted_parameter_ref[i][1:])
                
                explainer = shap.Explainer(self.tinnet_ead,
                                           inp_shap_ref)
                
                inp_shap_target = np.atleast_2d(predicted_parameter_target[i][1:])
                
                shap_values = explainer(inp_shap_target).values
                
                shap_vad2 += [shap_values[:,0]]
                shap_d_cen += [shap_values[:,1]]
                shap_width += [shap_values[:,2]]
                shap_adse_2 += [shap_values[:,3]]
                shap_beta_2 += [shap_values[:,4]]
                shap_delta_2 += [shap_values[:,5]]
                shap_adse_3 += [shap_values[:,6]]
                shap_beta_3 += [shap_values[:,7]]
                shap_delta_3 += [shap_values[:,8]]
                
                predicted_d_cen_ref += [predicted_parameter_ref[i][0]]
                predicted_d_cen_target += [predicted_parameter_target[i][0]]
            
            return np.vstack((np.array(predicted_d_cen_ref).flatten(),
                              np.array(predicted_d_cen_target).flatten(),
                              np.array(shap_vad2).flatten(),
                              np.array(shap_d_cen).flatten(),
                              np.array(shap_width).flatten(),
                              np.array(shap_adse_2).flatten(),
                              np.array(shap_beta_2).flatten(),
                              np.array(shap_delta_2).flatten(),
                              np.array(shap_adse_3).flatten(),
                              np.array(shap_beta_3).flatten(),
                              np.array(shap_delta_3).flatten())), parm_diff
    def tinnet_ead(self,
                   parm):
        parm = torch.Tensor(parm)
        
        h = np.zeros(3001)
        if 3001 % 2 == 0:
            h[0] = h[3001 // 2] = 1
            h[1:3001 // 2] = 2
        else:
            h[0] = 1
            h[1:(3001+1) // 2] = 2
        
        h = torch.Tensor(h)
        ergy = torch.Tensor(np.linspace(-15, 15, 3001))
        
        fermi = np.argsort(abs(ergy))[0] + 1
        
        if self.phys_model == 'OH_atop':
            vad2 = parm[:,0]
            d_cen = parm[:,1]
            width = parm[:,2]
            adse_1 = parm[:,3]
            beta_1 = parm[:,4]
            delta_1 = parm[:,5]
            adse_2 = parm[:,6]
            beta_2 = parm[:,7]
            delta_2 = parm[:,8]
            adse_3 = parm[:,9]
            beta_3 = parm[:,10]
            delta_3 = parm[:,11]
            
            # Semi-ellipse
            dos_d = (abs(1-((ergy[None,:]-d_cen[:,None])/width[:,None])**2))**0.5
            dos_d = dos_d * (abs(ergy[None,:]-d_cen[:,None]) < width[:,None])
            dos_d = dos_d + (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
            
            f = torch.trapz(dos_d[:,0:fermi],ergy[0:fermi])
            
            wdos_1 = np.pi * (beta_1[:,None]*vad2[:,None]*dos_d) + delta_1[:,None]
            wdos_1_ = np.pi * (0*vad2[:,None]*dos_d) + delta_1[:,None]
            wdos_2 = np.pi * (beta_2[:,None]*vad2[:,None]*dos_d) + delta_2[:,None]
            wdos_2_ = np.pi * (0*vad2[:,None]*dos_d) + delta_2[:,None]
            wdos_3 = np.pi * (beta_3[:,None]*vad2[:,None]*dos_d) + delta_3[:,None]
            wdos_3_ = np.pi * (0*vad2[:,None]*dos_d) + delta_3[:,None]
            
            eps = np.finfo(float).eps
            
            # Hilbert transform
            af_1 = torch.fft.fft(wdos_1, dim=1)
            htwdos_1 = torch.imag(torch.fft.ifft(af_1*h[None,:]))
            deno_1 = (ergy[None,:] - adse_1[:,None] - htwdos_1)
            deno_1 = deno_1 * (torch.abs(deno_1) > eps) + eps * (torch.abs(deno_1) <= eps) * (deno_1 >= 0) - eps * (torch.abs(deno_1) <= eps) * (deno_1 < 0)
            integrand_1 = wdos_1 / deno_1
            arctan_1 = torch.atan(integrand_1)
            arctan_1 = (arctan_1-np.pi)*(arctan_1 > 0) + (arctan_1)*(arctan_1 <= 0)
            d_hyb_1 = 2 / np.pi * torch.trapz(arctan_1[:,0:fermi],ergy[None,0:fermi])
            
            lorentzian_1 = (1/np.pi) * (delta_1[:,None])/((ergy[None,:] - adse_1[:,None])**2 + delta_1[:,None]**2)
            na_1 = torch.trapz(lorentzian_1[:,0:fermi], ergy[None,0:fermi])
            
            deno_1_ = (ergy[None,:] - adse_1[:,None])
            deno_1_ = deno_1_ * (torch.abs(deno_1_) > eps) + eps * (torch.abs(deno_1_) <= eps) * (deno_1_ >= 0) - eps * (torch.abs(deno_1_) <= eps) * (deno_1_ < 0)
            integrand_1_ = wdos_1_ / deno_1_
            arctan_1_ = torch.atan(integrand_1_)
            arctan_1_ = (arctan_1_-np.pi)*(arctan_1_ > 0) + (arctan_1_)*(arctan_1_ <= 0)
            d_hyb_1_ = 2 / np.pi * torch.trapz(arctan_1_[:,0:fermi],ergy[None,0:fermi])
            
            energy_NA_1 = d_hyb_1 - d_hyb_1_
            
            dos_ads_1 = wdos_1/(deno_1**2+wdos_1**2)/np.pi
            dos_ads_1 = dos_ads_1/torch.trapz(dos_ads_1, ergy[None,:])[:,None]
            
            af_2 = torch.fft.fft(wdos_2, dim=1)
            htwdos_2 = torch.imag(torch.fft.ifft(af_2*h[None,:]))
            deno_2 = (ergy[None,:] - adse_2[:,None] - htwdos_2)
            deno_2 = deno_2 * (torch.abs(deno_2) > eps) + eps * (torch.abs(deno_2) <= eps) * (deno_2 >= 0) - eps * (torch.abs(deno_2) <= eps) * (deno_2 < 0)
            integrand_2 = wdos_2 / deno_2
            arctan_2 = torch.atan(integrand_2)
            arctan_2 = (arctan_2-np.pi)*(arctan_2 > 0) + (arctan_2)*(arctan_2 <= 0)
            d_hyb_2 = 2 / np.pi * torch.trapz(arctan_2[:,0:fermi],ergy[None,0:fermi])
            
            lorentzian_2 = (1/np.pi) * (delta_2[:,None])/((ergy[None,:] - adse_2[:,None])**2 + delta_2[:,None]**2)
            na_2 = torch.trapz(lorentzian_2[:,0:fermi], ergy[None,0:fermi])
            
            deno_2_ = (ergy[None,:] - adse_2[:,None])
            deno_2_ = deno_2_ * (torch.abs(deno_2_) > eps) + eps * (torch.abs(deno_2_) <= eps) * (deno_2_ >= 0) - eps * (torch.abs(deno_2_) <= eps) * (deno_2_ < 0)
            integrand_2_ = wdos_2_ / deno_2_
            arctan_2_ = torch.atan(integrand_2_)
            arctan_2_ = (arctan_2_-np.pi)*(arctan_2_ > 0) + (arctan_2_)*(arctan_2_ <= 0)
            d_hyb_2_ = 2 / np.pi * torch.trapz(arctan_2_[:,0:fermi],ergy[None,0:fermi])
            
            energy_NA_2 = d_hyb_2 - d_hyb_2_
            
            dos_ads_2 = wdos_2/(deno_2**2+wdos_2**2)/np.pi
            dos_ads_2 = dos_ads_2/torch.trapz(dos_ads_2, ergy[None,:])[:,None]
            
            af_3 = torch.fft.fft(wdos_3, dim=1)
            htwdos_3 = torch.imag(torch.fft.ifft(af_3*h[None,:]))
            deno_3 = (ergy[None,:] - adse_3[:,None] - htwdos_3)
            deno_3 = deno_3 * (torch.abs(deno_3) > eps) + eps * (torch.abs(deno_3) <= eps) * (deno_3 >= 0) - eps * (torch.abs(deno_3) <= eps) * (deno_3 < 0)
            integrand_3 = wdos_3 / deno_3
            arctan_3 = torch.atan(integrand_3)
            arctan_3 = (arctan_3-np.pi)*(arctan_3 > 0) + (arctan_3)*(arctan_3 <= 0)
            d_hyb_3 = 2 / np.pi * torch.trapz(arctan_3[:,0:fermi],ergy[None,0:fermi])
            
            lorentzian_3 = (1/np.pi) * (delta_3[:,None])/((ergy[None,:] - adse_3[:,None])**2 + delta_3[:,None]**2)
            na_3 = torch.trapz(lorentzian_3[:,0:fermi], ergy[None,0:fermi])
            
            deno_3_ = (ergy[None,:] - adse_3[:,None])
            deno_3_ = deno_3_ * (torch.abs(deno_3_) > eps) + eps * (torch.abs(deno_3_) <= eps) * (deno_3_ >= 0) - eps * (torch.abs(deno_3_) <= eps) * (deno_3_ < 0)
            integrand_3_ = wdos_3_ / deno_3_
            arctan_3_ = torch.atan(integrand_3_)
            arctan_3_ = (arctan_3_-np.pi)*(arctan_3_ > 0) + (arctan_3_)*(arctan_3_ <= 0)
            d_hyb_3_ = 2 / np.pi * torch.trapz(arctan_3_[:,0:fermi],ergy[None,0:fermi])
            
            energy_NA_3 = d_hyb_3 - d_hyb_3_
            
            dos_ads_3 = wdos_3/(deno_3**2+wdos_3**2)/np.pi
            dos_ads_3 = dos_ads_3/torch.trapz(dos_ads_3, ergy[None,:])[:,None]
            
            esp = -2.693696878597913
            alpha = 0.06378761202273762
            
            energy = (esp
                      + (energy_NA_1 + 2*(na_1+f)*alpha*beta_1*vad2)
                      + (energy_NA_2 + 2*(na_2+f)*alpha*beta_2*vad2) * 2
                      + (energy_NA_3 + 2*(na_3+f)*alpha*beta_3*vad2))
            
            return np.atleast_1d(energy.detach().cpu().numpy())
            
        if self.phys_model == 'O_atop':
            vad2 = parm[:,0]
            d_cen = parm[:,1]
            width = parm[:,2]
            adse_2 = parm[:,3]
            beta_2 = parm[:,4]
            delta_2 = parm[:,5]
            adse_3 = parm[:,6]
            beta_3 = parm[:,7]
            delta_3 = parm[:,8]
            
            # Semi-ellipse
            dos_d = (abs(1-((ergy[None,:]-d_cen[:,None])/width[:,None])**2))**0.5
            dos_d = dos_d * (abs(ergy[None,:]-d_cen[:,None]) < width[:,None])
            dos_d = dos_d + (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
            
            f = torch.trapz(dos_d[:,0:fermi],ergy[0:fermi])
            
            wdos_2 = np.pi * (beta_2[:,None]*vad2[:,None]*dos_d) + delta_2[:,None]
            wdos_2_ = np.pi * (0*vad2[:,None]*dos_d) + delta_2[:,None]
            wdos_3 = np.pi * (beta_3[:,None]*vad2[:,None]*dos_d) + delta_3[:,None]
            wdos_3_ = np.pi * (0*vad2[:,None]*dos_d) + delta_3[:,None]
            
            eps = np.finfo(float).eps
            
            # Hilbert transform
            af_2 = torch.fft.fft(wdos_2, dim=1)
            htwdos_2 = torch.imag(torch.fft.ifft(af_2*h[None,:]))
            deno_2 = (ergy[None,:] - adse_2[:,None] - htwdos_2)
            deno_2 = deno_2 * (torch.abs(deno_2) > eps) + eps * (torch.abs(deno_2) <= eps) * (deno_2 >= 0) - eps * (torch.abs(deno_2) <= eps) * (deno_2 < 0)
            integrand_2 = wdos_2 / deno_2
            arctan_2 = torch.atan(integrand_2)
            arctan_2 = (arctan_2-np.pi)*(arctan_2 > 0) + (arctan_2)*(arctan_2 <= 0)
            d_hyb_2 = 2 / np.pi * torch.trapz(arctan_2[:,0:fermi],ergy[None,0:fermi])
            
            lorentzian_2 = (1/np.pi) * (delta_2[:,None])/((ergy[None,:] - adse_2[:,None])**2 + delta_2[:,None]**2)
            na_2 = torch.trapz(lorentzian_2[:,0:fermi], ergy[None,0:fermi])
            
            deno_2_ = (ergy[None,:] - adse_2[:,None])
            deno_2_ = deno_2_ * (torch.abs(deno_2_) > eps) + eps * (torch.abs(deno_2_) <= eps) * (deno_2_ >= 0) - eps * (torch.abs(deno_2_) <= eps) * (deno_2_ < 0)
            integrand_2_ = wdos_2_ / deno_2_
            arctan_2_ = torch.atan(integrand_2_)
            arctan_2_ = (arctan_2_-np.pi)*(arctan_2_ > 0) + (arctan_2_)*(arctan_2_ <= 0)
            d_hyb_2_ = 2 / np.pi * torch.trapz(arctan_2_[:,0:fermi],ergy[None,0:fermi])
            
            energy_NA_2 = d_hyb_2 - d_hyb_2_
            
            dos_ads_2 = wdos_2/(deno_2**2+wdos_2**2)/np.pi
            dos_ads_2 = dos_ads_2/torch.trapz(dos_ads_2, ergy[None,:])[:,None]
            
            af_3 = torch.fft.fft(wdos_3, dim=1)
            htwdos_3 = torch.imag(torch.fft.ifft(af_3*h[None,:]))
            deno_3 = (ergy[None,:] - adse_3[:,None] - htwdos_3)
            deno_3 = deno_3 * (torch.abs(deno_3) > eps) + eps * (torch.abs(deno_3) <= eps) * (deno_3 >= 0) - eps * (torch.abs(deno_3) <= eps) * (deno_3 < 0)
            integrand_3 = wdos_3 / deno_3
            arctan_3 = torch.atan(integrand_3)
            arctan_3 = (arctan_3-np.pi)*(arctan_3 > 0) + (arctan_3)*(arctan_3 <= 0)
            d_hyb_3 = 2 / np.pi * torch.trapz(arctan_3[:,0:fermi],ergy[None,0:fermi])
            
            lorentzian_3 = (1/np.pi) * (delta_3[:,None])/((ergy[None,:] - adse_3[:,None])**2 + delta_3[:,None]**2)
            na_3 = torch.trapz(lorentzian_3[:,0:fermi], ergy[None,0:fermi])
            
            deno_3_ = (ergy[None,:] - adse_3[:,None])
            deno_3_ = deno_3_ * (torch.abs(deno_3_) > eps) + eps * (torch.abs(deno_3_) <= eps) * (deno_3_ >= 0) - eps * (torch.abs(deno_3_) <= eps) * (deno_3_ < 0)
            integrand_3_ = wdos_3_ / deno_3_
            arctan_3_ = torch.atan(integrand_3_)
            arctan_3_ = (arctan_3_-np.pi)*(arctan_3_ > 0) + (arctan_3_)*(arctan_3_ <= 0)
            d_hyb_3_ = 2 / np.pi * torch.trapz(arctan_3_[:,0:fermi],ergy[None,0:fermi])
            
            energy_NA_3 = d_hyb_3 - d_hyb_3_
            
            dos_ads_3 = wdos_3/(deno_3**2+wdos_3**2)/np.pi
            dos_ads_3 = dos_ads_3/torch.trapz(dos_ads_3, ergy[None,:])[:,None]
            
            esp = -3.765294246337454
            alpha_2 = 0.07889181751783157
            alpha_3 = 0.05687347683456299
            
            energy = (esp
                      + (energy_NA_2 + 2*(na_2+f)*alpha_2*beta_2*vad2)
                      + (energy_NA_3 + 2*(na_3+f)*alpha_3*beta_3*vad2) * 2)
            
            return np.atleast_1d(energy.detach().cpu().numpy())
        
    def explain_shap(self,
                     ref_image=None,
                     ref_site_inx=None,
                     ref_name='Reference',
                     plot_name='shap',
                     save_fig='png'):
        
        target_image = self.image
        target_site_inx = self.site_inx
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
        
        shap, parm_diff = self.gen_shap(ref_image,
                                        ref_site_inx,
                                        target_image,
                                        target_site_inx)
        
        shap = np.average(shap, axis=1)
        
        idx = np.argsort(shap[2:])
        
        parm_diff = parm_diff[1:]
        
        if self.phys_model == 'OH_atop':
            labels = [r'$V_{ad}^{2}$',
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
                      r'$\delta_{4\sigma^{*}}$']
            
            labels = [labels[i] for i in idx]
            
            parm_diff = parm_diff[idx]
            
            idx = np.concatenate(([0,1], idx+2))
            shap = shap[idx]
            
            dx12 = shap[2]
            dx11 = shap[3]
            dx10 = shap[4]
            dx9 = shap[5]
            dx8 = shap[6]
            dx7 = shap[7]
            dx6 = shap[8]
            dx5 = shap[9]
            dx4 = shap[10]
            dx3 = shap[11]
            dx2 = shap[12]
            dx1 = shap[13]
            
            x12 = shap[0]
            x11 = x12 + dx12
            x10 = x11 + dx11
            x9 = x10 + dx10
            x8 = x9 + dx9
            x7 = x8 + dx8
            x6 = x7 + dx7
            x5 = x6 + dx6
            x4 = x5 + dx5
            x3 = x4 + dx4
            x2 = x3 + dx3
            x1 = x2 + dx2
            
            c12 = (shap[2] < 0) * 'red' + (shap[2] >= 0) * 'blue'
            c11 = (shap[3] < 0) * 'red' + (shap[3] >= 0) * 'blue'
            c10 = (shap[4] < 0) * 'red' + (shap[4] >= 0) * 'blue'
            c9 = (shap[5] < 0) * 'red' + (shap[5] >= 0) * 'blue'
            c8 = (shap[6] < 0) * 'red' + (shap[6] >= 0) * 'blue'
            c7 = (shap[7] < 0) * 'red' + (shap[7] >= 0) * 'blue'
            c6 = (shap[8] < 0) * 'red' + (shap[8] >= 0) * 'blue'
            c5 = (shap[9] < 0) * 'red' + (shap[9] >= 0) * 'blue'
            c4 = (shap[10] < 0) * 'red' + (shap[10] >= 0) * 'blue'
            c3 = (shap[11] < 0) * 'red' + (shap[11] >= 0) * 'blue'
            c2 = (shap[12] < 0) * 'red' + (shap[12] >= 0) * 'blue'
            c1 = (shap[13] < 0) * 'red' + (shap[13] >= 0) * 'blue'
            
            ax.arrow(x=x12, y=12, dx=dx12, dy=0, color=c12, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx12),
                     length_includes_head=True)
            ax.arrow(x=x11, y=11, dx=dx11, dy=0, color=c11, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx11),
                     length_includes_head=True)
            ax.arrow(x=x10, y=10, dx=dx10, dy=0, color=c10, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx10),
                     length_includes_head=True)
            ax.arrow(x=x9, y=9, dx=dx9, dy=0, color=c9, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx9),
                     length_includes_head=True)
            ax.arrow(x=x8, y=8, dx=dx8, dy=0, color=c8, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx8),
                     length_includes_head=True)
            ax.arrow(x=x7, y=7, dx=dx7, dy=0, color=c7, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx7),
                     length_includes_head=True)
            ax.arrow(x=x6, y=6, dx=dx6, dy=0, color=c6, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx6),
                     length_includes_head=True)
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
            
            ax.set_ylim([0.5, 12.5])
            
            plt.yticks([12,11,10,9,8,7,6,5,4,3,2,1], labels)
            
            sym_12 = (shap[2] < 0) * '-' + (shap[2] >= 0) * '+'
            sym_11 = (shap[3] < 0) * '-' + (shap[3] >= 0) * '+'
            sym_10 = (shap[4] < 0) * '-' + (shap[4] >= 0) * '+'
            sym_9 = (shap[5] < 0) * '-' + (shap[5] >= 0) * '+'
            sym_8 = (shap[6] < 0) * '-' + (shap[6] >= 0) * '+'
            sym_7 = (shap[7] < 0) * '-' + (shap[7] >= 0) * '+'
            sym_6 = (shap[8] < 0) * '-' + (shap[8] >= 0) * '+'
            sym_5 = (shap[9] < 0) * '-' + (shap[9] >= 0) * '+'
            sym_4 = (shap[10] < 0) * '-' + (shap[10] >= 0) * '+'
            sym_3 = (shap[11] < 0) * '-' + (shap[11] >= 0) * '+'
            sym_2 = (shap[12] < 0) * '-' + (shap[12] >= 0) * '+'
            sym_1 = (shap[13] < 0) * '-' + (shap[13] >= 0) * '+'
            
            ax.annotate(sym_12 + '{:.2f}'.format(round(np.abs(shap[2]), 4)),
                        xy=(1.12, 12.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c12)
            ax.annotate(sym_11 + '{:.2f}'.format(round(np.abs(shap[3]), 4)),
                        xy=(1.12, 11.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c11)
            ax.annotate(sym_10 + '{:.2f}'.format(round(np.abs(shap[4]), 4)),
                        xy=(1.12, 10.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c10)
            ax.annotate(sym_9 + '{:.2f}'.format(round(np.abs(shap[5]), 4)),
                        xy=(1.12, 9.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c9)
            ax.annotate(sym_8 + '{:.2f}'.format(round(np.abs(shap[6]), 4)),
                        xy=(1.12, 8.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c8)
            ax.annotate(sym_7 + '{:.2f}'.format(round(np.abs(shap[7]), 4)),
                        xy=(1.12, 7.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c7)
            ax.annotate(sym_6 + '{:.2f}'.format(round(np.abs(shap[8]), 4)),
                        xy=(1.12, 6.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c6)
            ax.annotate(sym_5 + '{:.2f}'.format(round(np.abs(shap[9]), 4)),
                        xy=(1.12, 5.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c5)
            ax.annotate(sym_4 + '{:.2f}'.format(round(np.abs(shap[10]), 4)),
                        xy=(1.12, 4.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c4)
            ax.annotate(sym_3 + '{:.2f}'.format(round(np.abs(shap[11]), 4)),
                        xy=(1.12, 3.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c3)
            ax.annotate(sym_2 + '{:.2f}'.format(round(np.abs(shap[12]), 4)),
                        xy=(1.12, 2.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c2)
            ax.annotate(sym_1 + '{:.2f}'.format(round(np.abs(shap[13]), 4)),
                        xy=(1.12, 1.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c1)
            
            c12 = (parm_diff[0] < 0) * 'red' + (parm_diff[0] >= 0) * 'blue'
            c11 = (parm_diff[1] < 0) * 'red' + (parm_diff[1] >= 0) * 'blue'
            c10 = (parm_diff[2] < 0) * 'red' + (parm_diff[2] >= 0) * 'blue'
            c9 = (parm_diff[3] < 0) * 'red' + (parm_diff[3] >= 0) * 'blue'
            c8 = (parm_diff[4] < 0) * 'red' + (parm_diff[4] >= 0) * 'blue'
            c7 = (parm_diff[5] < 0) * 'red' + (parm_diff[5] >= 0) * 'blue'
            c6 = (parm_diff[6] < 0) * 'red' + (parm_diff[6] >= 0) * 'blue'
            c5 = (parm_diff[7] < 0) * 'red' + (parm_diff[7] >= 0) * 'blue'
            c4 = (parm_diff[8] < 0) * 'red' + (parm_diff[8] >= 0) * 'blue'
            c3 = (parm_diff[9] < 0) * 'red' + (parm_diff[9] >= 0) * 'blue'
            c2 = (parm_diff[10] < 0) * 'red' + (parm_diff[10] >= 0) * 'blue'
            c1 = (parm_diff[11] < 0) * 'red' + (parm_diff[11] >= 0) * 'blue'
            
            sym_12 = (parm_diff[0] < 0) * '-' + (parm_diff[0] >= 0) * '+'
            sym_11 = (parm_diff[1] < 0) * '-' + (parm_diff[1] >= 0) * '+'
            sym_10 = (parm_diff[2] < 0) * '-' + (parm_diff[2] >= 0) * '+'
            sym_9 = (parm_diff[3] < 0) * '-' + (parm_diff[3] >= 0) * '+'
            sym_8 = (parm_diff[4] < 0) * '-' + (parm_diff[4] >= 0) * '+'
            sym_7 = (parm_diff[5] < 0) * '-' + (parm_diff[5] >= 0) * '+'
            sym_6 = (parm_diff[6] < 0) * '-' + (parm_diff[6] >= 0) * '+'
            sym_5 = (parm_diff[7] < 0) * '-' + (parm_diff[7] >= 0) * '+'
            sym_4 = (parm_diff[8] < 0) * '-' + (parm_diff[8] >= 0) * '+'
            sym_3 = (parm_diff[9] < 0) * '-' + (parm_diff[9] >= 0) * '+'
            sym_2 = (parm_diff[10] < 0) * '-' + (parm_diff[10] >= 0) * '+'
            sym_1 = (parm_diff[11] < 0) * '-' + (parm_diff[11] >= 0) * '+'
            
            ax.annotate(sym_12 + '{:.2f}'.format(round(np.abs(parm_diff[0]), 4)),
                        xy=(-0.12, 12.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c12)
            ax.annotate(sym_11 + '{:.2f}'.format(round(np.abs(parm_diff[1]), 4)),
                        xy=(-0.12, 11.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c11)
            ax.annotate(sym_10 + '{:.2f}'.format(round(np.abs(parm_diff[2]), 4)),
                        xy=(-0.12, 10.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c10)
            ax.annotate(sym_9 + '{:.2f}'.format(round(np.abs(parm_diff[3]), 4)),
                        xy=(-0.12, 9.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c9)
            ax.annotate(sym_8 + '{:.2f}'.format(round(np.abs(parm_diff[4]), 4)),
                        xy=(-0.12, 8.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c8)
            ax.annotate(sym_7 + '{:.2f}'.format(round(np.abs(parm_diff[5]), 4)),
                        xy=(-0.12, 7.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c7)
            ax.annotate(sym_6 + '{:.2f}'.format(round(np.abs(parm_diff[6]), 4)),
                        xy=(-0.12, 6.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c6)
            ax.annotate(sym_5 + '{:.2f}'.format(round(np.abs(parm_diff[7]), 4)),
                        xy=(-0.12, 5.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c5)
            ax.annotate(sym_4 + '{:.2f}'.format(round(np.abs(parm_diff[8]), 4)),
                        xy=(-0.12, 4.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c4)
            ax.annotate(sym_3 + '{:.2f}'.format(round(np.abs(parm_diff[9]), 4)),
                        xy=(-0.12, 3.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c3)
            ax.annotate(sym_2 + '{:.2f}'.format(round(np.abs(parm_diff[10]), 4)),
                        xy=(-0.12, 2.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c2)
            ax.annotate(sym_1 + '{:.2f}'.format(round(np.abs(parm_diff[11]), 4)),
                        xy=(-0.12, 1.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c1)
            
            ax.set_xlabel(r'$E_{ad}^{OH, top}$ (eV)')
            ax.spines[['left', 'right', 'top']].set_visible(False)
            
            ax.tick_params('y', length=0, width=0, which='major')
            
            ax.plot([x12, x12],[13, 0.5],'--', color='gray', linewidth=1)
            ax.plot([x11, x11], [12 + 1.0 / 3.0, 11 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x10, x10], [11 + 1.0 / 3.0, 10 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x9, x9], [10 + 1.0 / 3.0, 9 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x8, x8], [9 + 1.0 / 3.0, 8 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x7, x7], [8 + 1.0 / 3.0, 7 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x6, x6], [7 + 1.0 / 3.0, 6 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x5, x5], [6 + 1.0 / 3.0, 5 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x4, x4], [5 + 1.0 / 3.0, 4 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x3, x3], [4 + 1.0 / 3.0, 3 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x2, x2], [3 + 1.0 / 3.0, 2 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x1, x1], [2 + 1.0 / 3.0, 1 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            
            ax.plot([x1 + dx1, x1 + dx1],[14, 0.5],'--', color='orange',
                    linewidth=1)
            
            ax.annotate(ref_name + '\n{:.2f}'.format(round(x12, 4)),
                        xy=(x12, 1.15), xycoords=('data', 'axes fraction'),
                        ha='center', va='center', color='gray')
            ax.annotate(target_name + '\n{:.2f}'.format(round(x1 + dx1, 4)),
                        xy=(x1 + dx1, 1.05), xycoords=('data', 'axes fraction'),
                        ha='center', va='center', color='orange')
            
            ax.annotate('SHAP',
                        xy=(1.12, 1.05), xycoords=('axes fraction', 'axes fraction'),
                        ha='center', va='center', color='black')
            
            fig.tight_layout()
            
            if save_fig == 'png':
                fig.savefig(plot_name + '.png', bbox_inches='tight', dpi=600)
            elif save_fig == 'pdf':
                fig.savefig(plot_name + '.pdf', bbox_inches='tight')
        if self.phys_model == 'O_atop':
            labels = [r'$V_{ad}^{2}$',
                      r'$\epsilon_{d}$',
                      r'$W_{d}$',
                      r'$\epsilon_{pz}$',
                      r'$\beta_{pz}$',
                      r'$\delta_{pz}$',
                      r'$\epsilon_{pxy}$',
                      r'$\beta_{pxy}$',
                      r'$\delta_{pxy}$']
            
            labels = [labels[i] for i in idx]
            
            parm_diff = parm_diff[idx]
            
            idx = np.concatenate(([0,1], idx+2))
            shap = shap[idx]
            
            dx9 = shap[2]
            dx8 = shap[3]
            dx7 = shap[4]
            dx6 = shap[5]
            dx5 = shap[6]
            dx4 = shap[7]
            dx3 = shap[8]
            dx2 = shap[9]
            dx1 = shap[10]
            
            x9 = shap[0]
            x8 = x9 + dx9
            x7 = x8 + dx8
            x6 = x7 + dx7
            x5 = x6 + dx6
            x4 = x5 + dx5
            x3 = x4 + dx4
            x2 = x3 + dx3
            x1 = x2 + dx2
            
            c9 = (shap[2] < 0) * 'red' + (shap[2] >= 0) * 'blue'
            c8 = (shap[3] < 0) * 'red' + (shap[3] >= 0) * 'blue'
            c7 = (shap[4] < 0) * 'red' + (shap[4] >= 0) * 'blue'
            c6 = (shap[5] < 0) * 'red' + (shap[5] >= 0) * 'blue'
            c5 = (shap[6] < 0) * 'red' + (shap[6] >= 0) * 'blue'
            c4 = (shap[7] < 0) * 'red' + (shap[7] >= 0) * 'blue'
            c3 = (shap[8] < 0) * 'red' + (shap[8] >= 0) * 'blue'
            c2 = (shap[9] < 0) * 'red' + (shap[9] >= 0) * 'blue'
            c1 = (shap[10] < 0) * 'red' + (shap[10] >= 0) * 'blue'
            
            ax.arrow(x=x9, y=9, dx=dx9, dy=0, color=c9, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx9),
                     length_includes_head=True)
            ax.arrow(x=x8, y=8, dx=dx8, dy=0, color=c8, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx8),
                     length_includes_head=True)
            ax.arrow(x=x7, y=7, dx=dx7, dy=0, color=c7, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx7),
                     length_includes_head=True)
            ax.arrow(x=x6, y=6, dx=dx6, dy=0, color=c6, width=1.0/3.0,
                     head_width=1.0/3.0, head_length=0.15*np.abs(dx6),
                     length_includes_head=True)
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
            
            ax.set_ylim([0.5, 9.5])
            
            plt.yticks([9,8,7,6,5,4,3,2,1], labels)
            
            sym_9 = (shap[2] < 0) * '-' + (shap[2] >= 0) * '+'
            sym_8 = (shap[3] < 0) * '-' + (shap[3] >= 0) * '+'
            sym_7 = (shap[4] < 0) * '-' + (shap[4] >= 0) * '+'
            sym_6 = (shap[5] < 0) * '-' + (shap[5] >= 0) * '+'
            sym_5 = (shap[6] < 0) * '-' + (shap[6] >= 0) * '+'
            sym_4 = (shap[7] < 0) * '-' + (shap[7] >= 0) * '+'
            sym_3 = (shap[8] < 0) * '-' + (shap[8] >= 0) * '+'
            sym_2 = (shap[9] < 0) * '-' + (shap[9] >= 0) * '+'
            sym_1 = (shap[10] < 0) * '-' + (shap[10] >= 0) * '+'
            
            ax.annotate(sym_9 + '{:.2f}'.format(round(np.abs(shap[2]), 4)),
                        xy=(1.12, 9.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c9)
            ax.annotate(sym_8 + '{:.2f}'.format(round(np.abs(shap[3]), 4)),
                        xy=(1.12, 8.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c8)
            ax.annotate(sym_7 + '{:.2f}'.format(round(np.abs(shap[4]), 4)),
                        xy=(1.12, 7.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c7)
            ax.annotate(sym_6 + '{:.2f}'.format(round(np.abs(shap[5]), 4)),
                        xy=(1.12, 6.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c6)
            ax.annotate(sym_5 + '{:.2f}'.format(round(np.abs(shap[6]), 4)),
                        xy=(1.12, 5.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c5)
            ax.annotate(sym_4 + '{:.2f}'.format(round(np.abs(shap[7]), 4)),
                        xy=(1.12, 4.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c4)
            ax.annotate(sym_3 + '{:.2f}'.format(round(np.abs(shap[8]), 4)),
                        xy=(1.12, 3.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c3)
            ax.annotate(sym_2 + '{:.2f}'.format(round(np.abs(shap[9]), 4)),
                        xy=(1.12, 2.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c2)
            ax.annotate(sym_1 + '{:.2f}'.format(round(np.abs(shap[10]), 4)),
                        xy=(1.12, 1.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c1)
            
            c9 = (parm_diff[0] < 0) * 'red' + (parm_diff[0] >= 0) * 'blue'
            c8 = (parm_diff[1] < 0) * 'red' + (parm_diff[1] >= 0) * 'blue'
            c7 = (parm_diff[2] < 0) * 'red' + (parm_diff[2] >= 0) * 'blue'
            c6 = (parm_diff[3] < 0) * 'red' + (parm_diff[3] >= 0) * 'blue'
            c5 = (parm_diff[4] < 0) * 'red' + (parm_diff[4] >= 0) * 'blue'
            c4 = (parm_diff[5] < 0) * 'red' + (parm_diff[5] >= 0) * 'blue'
            c3 = (parm_diff[6] < 0) * 'red' + (parm_diff[6] >= 0) * 'blue'
            c2 = (parm_diff[7] < 0) * 'red' + (parm_diff[7] >= 0) * 'blue'
            c1 = (parm_diff[8] < 0) * 'red' + (parm_diff[8] >= 0) * 'blue'
            
            sym_9 = (parm_diff[0] < 0) * '-' + (parm_diff[0] >= 0) * '+'
            sym_8 = (parm_diff[1] < 0) * '-' + (parm_diff[1] >= 0) * '+'
            sym_7 = (parm_diff[2] < 0) * '-' + (parm_diff[2] >= 0) * '+'
            sym_6 = (parm_diff[3] < 0) * '-' + (parm_diff[3] >= 0) * '+'
            sym_5 = (parm_diff[4] < 0) * '-' + (parm_diff[4] >= 0) * '+'
            sym_4 = (parm_diff[5] < 0) * '-' + (parm_diff[5] >= 0) * '+'
            sym_3 = (parm_diff[6] < 0) * '-' + (parm_diff[6] >= 0) * '+'
            sym_2 = (parm_diff[7] < 0) * '-' + (parm_diff[7] >= 0) * '+'
            sym_1 = (parm_diff[8] < 0) * '-' + (parm_diff[8] >= 0) * '+'
            
            ax.annotate(sym_9 + '{:.2f}'.format(round(np.abs(parm_diff[0]), 4)),
                        xy=(-0.12, 9.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c9)
            ax.annotate(sym_8 + '{:.2f}'.format(round(np.abs(parm_diff[1]), 4)),
                        xy=(-0.12, 8.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c8)
            ax.annotate(sym_7 + '{:.2f}'.format(round(np.abs(parm_diff[2]), 4)),
                        xy=(-0.12, 7.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c7)
            ax.annotate(sym_6 + '{:.2f}'.format(round(np.abs(parm_diff[3]), 4)),
                        xy=(-0.12, 6.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c6)
            ax.annotate(sym_5 + '{:.2f}'.format(round(np.abs(parm_diff[4]), 4)),
                        xy=(-0.12, 5.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c5)
            ax.annotate(sym_4 + '{:.2f}'.format(round(np.abs(parm_diff[5]), 4)),
                        xy=(-0.12, 4.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c4)
            ax.annotate(sym_3 + '{:.2f}'.format(round(np.abs(parm_diff[6]), 4)),
                        xy=(-0.12, 3.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c3)
            ax.annotate(sym_2 + '{:.2f}'.format(round(np.abs(parm_diff[7]), 4)),
                        xy=(-0.12, 2.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c2)
            ax.annotate(sym_1 + '{:.2f}'.format(round(np.abs(parm_diff[8]), 4)),
                        xy=(-0.12, 1.00), xycoords=('axes fraction', 'data'),
                        ha='center', va='center', color=c1)
            
            ax.set_xlabel(r'$E_{ad}^{O, top}$ (eV)')
            ax.spines[['left', 'right', 'top']].set_visible(False)
            
            ax.tick_params('y', length=0, width=0, which='major')
            
            ax.plot([x9, x9],[11, 0.5],'--', color='gray', linewidth=1)
            
            ax.plot([x8, x8], [9 + 1.0 / 3.0, 8 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x7, x7], [8 + 1.0 / 3.0, 7 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x6, x6], [7 + 1.0 / 3.0, 6 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x5, x5], [6 + 1.0 / 3.0, 5 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x4, x4], [5 + 1.0 / 3.0, 4 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x3, x3], [4 + 1.0 / 3.0, 3 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x2, x2], [3 + 1.0 / 3.0, 2 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            ax.plot([x1, x1], [2 + 1.0 / 3.0, 1 - 1.0 / 3.0],'--', color='gray',
                    linewidth=1)
            
            ax.plot([x1 + dx1, x1 + dx1],[11, 0.5],'--', color='orange',
                    linewidth=1)
            
            ax.annotate(ref_name + '\n{:.2f}'.format(round(x9, 4)),
                        xy=(x9, 1.15), xycoords=('data', 'axes fraction'),
                        ha='center', va='center', color='gray')
            ax.annotate(target_name + '\n{:.2f}'.format(round(x1 + dx1, 4)),
                        xy=(x1 + dx1, 1.05), xycoords=('data', 'axes fraction'),
                        ha='center', va='center', color='orange')
            
            ax.annotate('SHAP',
                        xy=(1.12, 1.05), xycoords=('axes fraction', 'axes fraction'),
                        ha='center', va='center', color='black')
            
            fig.tight_layout()
            
            if save_fig == 'png':
                fig.savefig(plot_name + '.png', bbox_inches='tight', dpi=600)
            elif save_fig == 'pdf':
                fig.savefig(plot_name + '.pdf', bbox_inches='tight')

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
        
        dataset = [((torch.Tensor(atom_fea),
                     torch.Tensor(nbr_fea),
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
    
    def eval_model(self, **kwargs):
        if self.phys_model == 'OH_atop':
            if self.cuda:
                best_checkpoint = torch.load('./data/pretrained/adsorption_energy/OH/atop/model_' + str(self.model_inx) + '.pth.tar')
            else:
                best_checkpoint = torch.load('./data/pretrained/adsorption_energy/OH/atop/model_' + str(self.model_inx) + '.pth.tar', map_location=torch.device('cpu'))
            
            self.model.load_state_dict(best_checkpoint['state_dict'])
            
            # switch to evaluate mode
            self.model.eval()
            
            for i, (input, batch_cif_ids) in enumerate(self.data_loader):
                with torch.no_grad():
                    if self.cuda:
                        input_var = (Variable(input[0].cuda(non_blocking=True)),
                                     Variable(input[1].cuda(non_blocking=True)),
                                     input[2].cuda(non_blocking=True),
                                     [crys_idx.cuda(non_blocking=True)
                                      for crys_idx in input[3]],
                                     [site_inx.cuda(non_blocking=True)
                                      for site_inx in input[4]])
                    else:
                        input_var = (Variable(input[0]),
                                     Variable(input[1]),
                                     input[2],
                                     input[3],
                                     input[4])
                
                # compute output
                cnn_output = self.model(*input_var)
                
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    phys_model=self.phys_model,
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            return output, parm
        if self.phys_model == 'O_atop':
            if self.cuda:
                best_checkpoint = torch.load('./data/pretrained/adsorption_energy/O/atop/model_' + str(self.model_inx) + '.pth.tar')
            else:
                best_checkpoint = torch.load('./data/pretrained/adsorption_energy/O/atop/model_' + str(self.model_inx) + '.pth.tar', map_location=torch.device('cpu'))
            
            self.model.load_state_dict(best_checkpoint['state_dict'])
            
            # switch to evaluate mode
            self.model.eval()
            
            for i, (input, batch_cif_ids) in enumerate(self.data_loader):
                with torch.no_grad():
                    if self.cuda:
                        input_var = (Variable(input[0].cuda(non_blocking=True)),
                                     Variable(input[1].cuda(non_blocking=True)),
                                     input[2].cuda(non_blocking=True),
                                     [crys_idx.cuda(non_blocking=True)
                                      for crys_idx in input[3]],
                                     [site_inx.cuda(non_blocking=True)
                                      for site_inx in input[4]])
                    else:
                        input_var = (Variable(input[0]),
                                     Variable(input[1]),
                                     input[2],
                                     input[3],
                                     input[4])
                
                # compute output
                cnn_output = self.model(*input_var)
                
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    phys_model=self.phys_model,
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            return output, parm
    
    def get_data_loader(self,
                        dataset,
                        collate_fn=default_collate,
                        batch_size=256,
                        num_workers=0,
                        pin_memory=False,
                        random_seed=None):
        
        data_sampler = SubsetRandomSampler(np.arange(len(dataset)))
        
        data_loader = DataLoader(dataset, batch_size=1024,
                                 sampler=data_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)
        
        return data_loader

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

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        '''
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class Chemisorption:
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
        namodel_in = torch.flatten(namodel_in)
        vad2 = self.vad2
        h = self.h
        ergy = self.ergy
        fermi = np.argsort(abs(ergy.detach().cpu().numpy()))[0] + 1
        eps = np.finfo(float).eps
        
        if phys_model == 'OH_atop':
            adse_1 = namodel_in[0]
            beta_1 = torch.nn.functional.softplus(namodel_in[1])
            delta_1 = torch.nn.functional.softplus(namodel_in[2])
            adse_2 = namodel_in[3]
            beta_2 = torch.nn.functional.softplus(namodel_in[4])
            delta_2 = torch.nn.functional.softplus(namodel_in[5])
            adse_3 = namodel_in[6]
            beta_3 = torch.nn.functional.softplus(namodel_in[7])
            delta_3 = torch.nn.functional.softplus(namodel_in[8])
            d_cen = namodel_in[9]
            width = torch.nn.functional.softplus(namodel_in[10])
            
            # Semi-ellipse
            dos_d = (abs(1-((ergy-d_cen)/width)**2))**0.5
            dos_d = dos_d * (abs(ergy-d_cen) < width)
            dos_d = dos_d + (torch.trapz(dos_d,ergy) <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)
            
            f = torch.trapz(dos_d[0:fermi],ergy[0:fermi])
            
            wdos_1 = np.pi * (beta_1*vad2*dos_d) + delta_1
            wdos_1_ = np.pi * (0*vad2*dos_d) + delta_1
            wdos_2 = np.pi * (beta_2*vad2*dos_d) + delta_2
            wdos_2_ = np.pi * (0*vad2*dos_d) + delta_2
            wdos_3 = np.pi * (beta_3*vad2*dos_d) + delta_3
            wdos_3_ = np.pi * (0*vad2*dos_d) + delta_3
            
            # Hilbert transform
            af_1 = torch.fft.fft(wdos_1)
            htwdos_1 = torch.imag(torch.fft.ifft(af_1*h))
            deno_1 = (ergy - adse_1 - htwdos_1)
            deno_1 = deno_1 * (torch.abs(deno_1) > eps) + eps * (torch.abs(deno_1) <= eps) * (deno_1 >= 0) - eps * (torch.abs(deno_1) <= eps) * (deno_1 < 0)
            integrand_1 = wdos_1 / deno_1
            arctan_1 = torch.atan(integrand_1)
            arctan_1 = (arctan_1-np.pi)*(arctan_1 > 0) + (arctan_1)*(arctan_1 <= 0)
            d_hyb_1 = 2 / np.pi * torch.trapz(arctan_1[0:fermi],ergy[0:fermi])
            
            lorentzian_1 = (1/np.pi) * (delta_1)/((ergy - adse_1)**2 + delta_1**2)
            na_1 = torch.trapz(lorentzian_1[0:fermi], ergy[0:fermi])
            
            deno_1_ = (ergy - adse_1)
            deno_1_ = deno_1_ * (torch.abs(deno_1_) > eps) + eps * (torch.abs(deno_1_) <= eps) * (deno_1_ >= 0) - eps * (torch.abs(deno_1_) <= eps) * (deno_1_ < 0)
            integrand_1_ = wdos_1_ / deno_1_
            arctan_1_ = torch.atan(integrand_1_)
            arctan_1_ = (arctan_1_-np.pi)*(arctan_1_ > 0) + (arctan_1_)*(arctan_1_ <= 0)
            d_hyb_1_ = 2 / np.pi * torch.trapz(arctan_1_[0:fermi],ergy[0:fermi])
            
            energy_NA_1 = d_hyb_1 - d_hyb_1_
            
            dos_ads_1 = wdos_1/(deno_1**2+wdos_1**2)/np.pi
            dos_ads_1 = dos_ads_1/torch.trapz(dos_ads_1, ergy)
            
            af_2 = torch.fft.fft(wdos_2)
            htwdos_2 = torch.imag(torch.fft.ifft(af_2*h))
            deno_2 = (ergy - adse_2 - htwdos_2)
            deno_2 = deno_2 * (torch.abs(deno_2) > eps) + eps * (torch.abs(deno_2) <= eps) * (deno_2 >= 0) - eps * (torch.abs(deno_2) <= eps) * (deno_2 < 0)
            integrand_2 = wdos_2 / deno_2
            arctan_2 = torch.atan(integrand_2)
            arctan_2 = (arctan_2-np.pi)*(arctan_2 > 0) + (arctan_2)*(arctan_2 <= 0)
            d_hyb_2 = 2 / np.pi * torch.trapz(arctan_2[0:fermi],ergy[0:fermi])
            
            lorentzian_2 = (1/np.pi) * (delta_2)/((ergy - adse_2)**2 + delta_2**2)
            na_2 = torch.trapz(lorentzian_2[0:fermi], ergy[0:fermi])
            
            deno_2_ = (ergy - adse_2)
            deno_2_ = deno_2_ * (torch.abs(deno_2_) > eps) + eps * (torch.abs(deno_2_) <= eps) * (deno_2_ >= 0) - eps * (torch.abs(deno_2_) <= eps) * (deno_2_ < 0)
            integrand_2_ = wdos_2_ / deno_2_
            arctan_2_ = torch.atan(integrand_2_)
            arctan_2_ = (arctan_2_-np.pi)*(arctan_2_ > 0) + (arctan_2_)*(arctan_2_ <= 0)
            d_hyb_2_ = 2 / np.pi * torch.trapz(arctan_2_[0:fermi],ergy[0:fermi])
            
            energy_NA_2 = d_hyb_2 - d_hyb_2_
            
            dos_ads_2 = wdos_2/(deno_2**2+wdos_2**2)/np.pi
            dos_ads_2 = dos_ads_2/torch.trapz(dos_ads_2, ergy)
            
            af_3 = torch.fft.fft(wdos_3)
            htwdos_3 = torch.imag(torch.fft.ifft(af_3*h))
            deno_3 = (ergy - adse_3 - htwdos_3)
            deno_3 = deno_3 * (torch.abs(deno_3) > eps) + eps * (torch.abs(deno_3) <= eps) * (deno_3 >= 0) - eps * (torch.abs(deno_3) <= eps) * (deno_3 < 0)
            integrand_3 = wdos_3 / deno_3
            arctan_3 = torch.atan(integrand_3)
            arctan_3 = (arctan_3-np.pi)*(arctan_3 > 0) + (arctan_3)*(arctan_3 <= 0)
            d_hyb_3 = 2 / np.pi * torch.trapz(arctan_3[0:fermi],ergy[0:fermi])
            
            lorentzian_3 = (1/np.pi) * (delta_3)/((ergy - adse_3)**2 + delta_3**2)
            na_3 = torch.trapz(lorentzian_3[0:fermi], ergy[0:fermi])
            
            deno_3_ = (ergy - adse_3)
            deno_3_ = deno_3_ * (torch.abs(deno_3_) > eps) + eps * (torch.abs(deno_3_) <= eps) * (deno_3_ >= 0) - eps * (torch.abs(deno_3_) <= eps) * (deno_3_ < 0)
            integrand_3_ = wdos_3_ / deno_3_
            arctan_3_ = torch.atan(integrand_3_)
            arctan_3_ = (arctan_3_-np.pi)*(arctan_3_ > 0) + (arctan_3_)*(arctan_3_ <= 0)
            d_hyb_3_ = 2 / np.pi * torch.trapz(arctan_3_[0:fermi],ergy[0:fermi])
            
            energy_NA_3 = d_hyb_3 - d_hyb_3_
            
            dos_ads_3 = wdos_3/(deno_3**2+wdos_3**2)/np.pi
            dos_ads_3 = dos_ads_3/torch.trapz(dos_ads_3, ergy)
            
            energy = (self.esp
                      + (energy_NA_1 + 2*(na_1+f)*self.alpha*beta_1*vad2)
                      + (energy_NA_2 + 2*(na_2+f)*self.alpha*beta_2*vad2) * 2
                      + (energy_NA_3 + 2*(na_3+f)*self.alpha*beta_3*vad2))
            
            parm = torch.Tensor((energy[0],
                                 vad2[0],
                                 d_cen,
                                 width,
                                 adse_1,
                                 beta_1,
                                 delta_1,
                                 adse_2,
                                 beta_2,
                                 delta_2,
                                 adse_3,
                                 beta_3,
                                 delta_3))
            
            return energy.detach().cpu().numpy(), parm
        if phys_model == 'O_atop':
            adse_2 = namodel_in[0]
            beta_2 = torch.nn.functional.softplus(namodel_in[1])
            delta_2 = torch.nn.functional.softplus(namodel_in[2])
            adse_3 = namodel_in[3]
            beta_3 = torch.nn.functional.softplus(namodel_in[4])
            delta_3 = torch.nn.functional.softplus(namodel_in[5])
            
            d_cen = self.d_cen
            width = self.half_width
            
            # Semi-ellipse
            dos_d = (abs(1-((ergy-d_cen)/width)**2))**0.5
            dos_d = dos_d * (abs(ergy-d_cen) < width)
            dos_d = dos_d + (torch.trapz(dos_d,ergy) <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)
            
            f = torch.trapz(dos_d[0:fermi],ergy[0:fermi])
            
            wdos_2 = np.pi * (beta_2*vad2*dos_d) + delta_2
            wdos_2_ = np.pi * (0*vad2*dos_d) + delta_2
            wdos_3 = np.pi * (beta_3*vad2*dos_d) + delta_3
            wdos_3_ = np.pi * (0*vad2*dos_d) + delta_3
            
            # Hilbert transform
            af_2 = torch.fft.fft(wdos_2)
            htwdos_2 = torch.imag(torch.fft.ifft(af_2*h))
            deno_2 = (ergy - adse_2 - htwdos_2)
            deno_2 = deno_2 * (torch.abs(deno_2) > eps) + eps * (torch.abs(deno_2) <= eps) * (deno_2 >= 0) - eps * (torch.abs(deno_2) <= eps) * (deno_2 < 0)
            integrand_2 = wdos_2 / deno_2
            arctan_2 = torch.atan(integrand_2)
            arctan_2 = (arctan_2-np.pi)*(arctan_2 > 0) + (arctan_2)*(arctan_2 <= 0)
            d_hyb_2 = 2 / np.pi * torch.trapz(arctan_2[0:fermi],ergy[0:fermi])
            
            lorentzian_2 = (1/np.pi) * (delta_2)/((ergy - adse_2)**2 + delta_2**2)
            na_2 = torch.trapz(lorentzian_2[0:fermi], ergy[0:fermi])
            
            deno_2_ = (ergy - adse_2)
            deno_2_ = deno_2_ * (torch.abs(deno_2_) > eps) + eps * (torch.abs(deno_2_) <= eps) * (deno_2_ >= 0) - eps * (torch.abs(deno_2_) <= eps) * (deno_2_ < 0)
            integrand_2_ = wdos_2_ / deno_2_
            arctan_2_ = torch.atan(integrand_2_)
            arctan_2_ = (arctan_2_-np.pi)*(arctan_2_ > 0) + (arctan_2_)*(arctan_2_ <= 0)
            d_hyb_2_ = 2 / np.pi * torch.trapz(arctan_2_[0:fermi],ergy[0:fermi])
            
            energy_NA_2 = d_hyb_2 - d_hyb_2_
            
            dos_ads_2 = wdos_2/(deno_2**2+wdos_2**2)/np.pi
            dos_ads_2 = dos_ads_2/torch.trapz(dos_ads_2, ergy)
            
            af_3 = torch.fft.fft(wdos_3)
            htwdos_3 = torch.imag(torch.fft.ifft(af_3*h))
            deno_3 = (ergy - adse_3 - htwdos_3)
            deno_3 = deno_3 * (torch.abs(deno_3) > eps) + eps * (torch.abs(deno_3) <= eps) * (deno_3 >= 0) - eps * (torch.abs(deno_3) <= eps) * (deno_3 < 0)
            integrand_3 = wdos_3 / deno_3
            arctan_3 = torch.atan(integrand_3)
            arctan_3 = (arctan_3-np.pi)*(arctan_3 > 0) + (arctan_3)*(arctan_3 <= 0)
            d_hyb_3 = 2 / np.pi * torch.trapz(arctan_3[0:fermi],ergy[0:fermi])
            
            lorentzian_3 = (1/np.pi) * (delta_3)/((ergy - adse_3)**2 + delta_3**2)
            na_3 = torch.trapz(lorentzian_3[0:fermi], ergy[0:fermi])
            
            deno_3_ = (ergy - adse_3)
            deno_3_ = deno_3_ * (torch.abs(deno_3_) > eps) + eps * (torch.abs(deno_3_) <= eps) * (deno_3_ >= 0) - eps * (torch.abs(deno_3_) <= eps) * (deno_3_ < 0)
            integrand_3_ = wdos_3_ / deno_3_
            arctan_3_ = torch.atan(integrand_3_)
            arctan_3_ = (arctan_3_-np.pi)*(arctan_3_ > 0) + (arctan_3_)*(arctan_3_ <= 0)
            d_hyb_3_ = 2 / np.pi * torch.trapz(arctan_3_[0:fermi],ergy[0:fermi])
            
            energy_NA_3 = d_hyb_3 - d_hyb_3_
            
            dos_ads_3 = wdos_3/(deno_3**2+wdos_3**2)/np.pi
            dos_ads_3 = dos_ads_3/torch.trapz(dos_ads_3, ergy)
            
            energy = (self.esp
                      + (energy_NA_2 + 2*(na_2+f)*self.alpha_2*beta_2*vad2)
                      + (energy_NA_3 + 2*(na_3+f)*self.alpha_3*beta_3*vad2) * 2)
            
            parm = torch.Tensor((energy[0],
                                 vad2[0],
                                 d_cen,
                                 width,
                                 adse_2,
                                 beta_2,
                                 delta_2,
                                 adse_3,
                                 beta_3,
                                 delta_3))
            
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
            self.dict_atom_fea = self.dict_atom_fea_default()
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
    
    def dict_atom_prop_default(self):
        atom_prop_dict = {'Ca': {'f': 0.1, 'vad2': 20.8 , 'eps_d':  None},
                          'Sc': {'f': 0.2, 'vad2':  7.90, 'eps_d':  None},
                          'Ti': {'f': 0.3, 'vad2':  4.65, 'eps_d':  1.50},
                          'V' : {'f': 0.4, 'vad2':  3.15, 'eps_d':  1.06},
                          'Cr': {'f': 0.5, 'vad2':  2.35, 'eps_d':  0.16},
                          'Mn': {'f': 0.6, 'vad2':  1.94, 'eps_d':  0.07},
                          'Fe': {'f': 0.7, 'vad2':  1.59, 'eps_d': -0.92},
                          'Co': {'f': 0.8, 'vad2':  1.34, 'eps_d': -1.17},
                          'Ni': {'f': 0.9, 'vad2':  1.16, 'eps_d': -1.29},
                          'Cu': {'f': 1.0, 'vad2':  1.00, 'eps_d': -2.67},
                          'Zn': {'f': 1.0, 'vad2':  0.46, 'eps_d':  None},
                          'Sr': {'f': 0.1, 'vad2': 36.5 , 'eps_d':  None},
                          'Y' : {'f': 0.2, 'vad2': 17.3 , 'eps_d':  None},
                          'Zr': {'f': 0.3, 'vad2': 10.90, 'eps_d':  1.95},
                          'Nb': {'f': 0.4, 'vad2':  7.73, 'eps_d':  1.41},
                          'Mo': {'f': 0.5, 'vad2':  6.62, 'eps_d':  0.35},
                          'Tc': {'f': 0.6, 'vad2':  4.71, 'eps_d': -0.60},
                          'Ru': {'f': 0.7, 'vad2':  3.87, 'eps_d': -1.41},
                          'Rh': {'f': 0.8, 'vad2':  3.32, 'eps_d': -1.73},
                          'Pd': {'f': 0.9, 'vad2':  2.78, 'eps_d': -1.83},
                          'Ag': {'f': 1.0, 'vad2':  2.26, 'eps_d': -4.30},
                          'Cd': {'f': 1.0, 'vad2':  1.58, 'eps_d':  None},
                          'Ba': {'f': 0.1, 'vad2': 41.5 , 'eps_d':  None},
                          'Lu': {'f': 0.2, 'vad2': 17.1 , 'eps_d':  None},
                          'Hf': {'f': 0.3, 'vad2': 11.90, 'eps_d':  2.47},
                          'Ta': {'f': 0.4, 'vad2':  9.05, 'eps_d':  2.00},
                          'W' : {'f': 0.5, 'vad2':  7.27, 'eps_d':  0.77},
                          'Re': {'f': 0.6, 'vad2':  6.04, 'eps_d': -0.51},
                          'Os': {'f': 0.7, 'vad2':  5.13, 'eps_d':  None},
                          'Ir': {'f': 0.8, 'vad2':  4.45, 'eps_d': -2.11},
                          'Pt': {'f': 0.9, 'vad2':  3.90, 'eps_d': -2.25},
                          'Au': {'f': 1.0, 'vad2':  3.35, 'eps_d': -3.56},
                          'Hg': {'f': 1.0, 'vad2':  2.64, 'eps_d':  None}}
        return atom_prop_dict
    
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
