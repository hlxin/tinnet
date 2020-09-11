'''
collection of chemisorption models.

newns_anderson:
'''

import torch
import numpy as np


class Chemisorption:

    def __init__(self, model_name, **kwargs):
        # Initialize the class
        if model_name == 'gcnn':
            self.model_num_input = 1
        if model_name == 'newns_anderson_semi':
            self.model_num_input = 14
    
    def newns_anderson_semi(self, namodel_in, model, **kwargs):
        
        adse_1 = namodel_in[:,0]
        alpha_1 = torch.sigmoid(namodel_in[:,1])
        beta_1 = torch.nn.functional.softplus(namodel_in[:,2])
        delta_1 = torch.nn.functional.softplus(namodel_in[:,3])
        adse_2 = namodel_in[:,4]
        alpha_2 = torch.sigmoid(namodel_in[:,5])
        beta_2 = torch.nn.functional.softplus(namodel_in[:,6])
        delta_2 = torch.nn.functional.softplus(namodel_in[:,7])
        adse_3 = namodel_in[:,8]
        alpha_3 = torch.sigmoid(namodel_in[:,9])
        beta_3 = torch.nn.functional.softplus(namodel_in[:,10])
        delta_3 = torch.nn.functional.softplus(namodel_in[:,11])
        d_cen = namodel_in[:,12]
        width = torch.nn.functional.softplus(namodel_in[:,13])
        
        idx = kwargs['batch_cif_ids']
        
        energy_DFT = self.energy[idx]
        vad2 = self.vad2[idx]
        d_cen_DFT = self.d_cen[idx]
        width_DFT = self.width[idx]
        dos_ads_1_DFT = self.dos_ads_1[idx]
        dos_ads_2_DFT = self.dos_ads_2[idx]
        dos_ads_3_DFT = self.dos_ads_3[idx]
        
        h = self.h
        ergy = self.ergy
        
        fermi = np.argsort(abs(ergy.detach().cpu().numpy()))[0] + 1
        
        # Semi-ellipse
        if model == 'dft':
            dos_d = (abs(1-((ergy[None,:]-d_cen_DFT[:,None])/width_DFT[:,None])**2))**0.5
            dos_d = dos_d * (abs(ergy[None,:]-d_cen_DFT[:,None]) < width_DFT[:,None])
            dos_d = dos_d + (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
        else:
            dos_d = (abs(1-((ergy[None,:]-d_cen[:,None])/width[:,None])**2))**0.5
            dos_d = dos_d * (abs(ergy[None,:]-d_cen[:,None]) < width[:,None])
            dos_d = dos_d + (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
        
        f = torch.trapz(dos_d[:,0:fermi],ergy[0:fermi])
        
        wdos_1 = np.pi * (beta_1[:,None]*vad2[:,None]*dos_d) + delta_1[:,None]
        wdos_2 = np.pi * (beta_2[:,None]*vad2[:,None]*dos_d) + delta_2[:,None]
        wdos_3 = np.pi * (beta_3[:,None]*vad2[:,None]*dos_d) + delta_3[:,None]
        
        eps = np.finfo(float).eps
        
        # Hilbert transform
        af_1 = torch.rfft(wdos_1,1,onesided=False)
        htwdos_1 = torch.ifft(af_1*h[None,:,None],1)[:,:,1]
        deno_1 = (ergy[None,:] - adse_1[:,None] - htwdos_1)
        deno_1 = deno_1 * (torch.abs(deno_1) > eps) + eps * (torch.abs(deno_1) <= eps) * (deno_1 >= 0) - eps * (torch.abs(deno_1) <= eps) * (deno_1 < 0)
        integrand_1 = wdos_1 / deno_1
        arctan_1 = torch.atan(integrand_1)
        arctan_1 = (arctan_1-np.pi)*(arctan_1 > 0) + (arctan_1)*(arctan_1 <= 0)
        d_hyb_1 = 2 / np.pi * torch.trapz(arctan_1[:,0:fermi],ergy[None,0:fermi])
        lorentzian_1 = (1/np.pi) * (delta_1[:,None])/((ergy[None,:] - adse_1[:,None])**2 + delta_1[:,None]**2)
        na_1 = torch.trapz(lorentzian_1[:,0:fermi], ergy[None,0:fermi])
        AvgVal_1 = torch.trapz(lorentzian_1[:,0:fermi] * ergy[None,0:fermi], ergy[None,0:fermi])
        energy_NA_1 = d_hyb_1 - 2*AvgVal_1
        dos_ads_1 = wdos_1/(deno_1**2+wdos_1**2)/np.pi
        dos_ads_1 = dos_ads_1/torch.trapz(dos_ads_1, ergy[None,:])[:,None]
        
        af_2 = torch.rfft(wdos_2,1,onesided=False)
        htwdos_2 = torch.ifft(af_2*h[None,:,None],1)[:,:,1]
        deno_2 = (ergy[None,:] - adse_2[:,None] - htwdos_2)
        deno_2 = deno_2 * (torch.abs(deno_2) > eps) + eps * (torch.abs(deno_2) <= eps) * (deno_2 >= 0) - eps * (torch.abs(deno_2) <= eps) * (deno_2 < 0)
        integrand_2 = wdos_2 / deno_2
        arctan_2 = torch.atan(integrand_2)
        arctan_2 = (arctan_2-np.pi)*(arctan_2 > 0) + (arctan_2)*(arctan_2 <= 0)
        d_hyb_2 = 2 / np.pi * torch.trapz(arctan_2[:,0:fermi],ergy[None,0:fermi])
        lorentzian_2 = (1/np.pi) * (delta_2[:,None])/((ergy[None,:] - adse_2[:,None])**2 + delta_2[:,None]**2)
        na_2 = torch.trapz(lorentzian_2[:,0:fermi], ergy[None,0:fermi])
        AvgVal_2 = torch.trapz(lorentzian_2[:,0:fermi] * ergy[None,0:fermi], ergy[None,0:fermi])
        energy_NA_2 = d_hyb_2 - 2*AvgVal_2
        dos_ads_2 = wdos_2/(deno_2**2+wdos_2**2)/np.pi
        dos_ads_2 = dos_ads_2/torch.trapz(dos_ads_2, ergy[None,:])[:,None]
        
        af_3 = torch.rfft(wdos_3,1,onesided=False)
        htwdos_3 = torch.ifft(af_3*h[None,:,None],1)[:,:,1]
        deno_3 = (ergy[None,:] - adse_3[:,None] - htwdos_3)
        deno_3 = deno_3 * (torch.abs(deno_3) > eps) + eps * (torch.abs(deno_3) <= eps) * (deno_3 >= 0) - eps * (torch.abs(deno_3) <= eps) * (deno_3 < 0)
        integrand_3 = wdos_3 / deno_3
        arctan_3 = torch.atan(integrand_3)
        arctan_3 = (arctan_3-np.pi)*(arctan_3 > 0) + (arctan_3)*(arctan_3 <= 0)
        d_hyb_3 = 2 / np.pi * torch.trapz(arctan_3[:,0:fermi],ergy[None,0:fermi])
        lorentzian_3 = (1/np.pi) * (delta_3[:,None])/((ergy[None,:] - adse_3[:,None])**2 + delta_3[:,None]**2)
        na_3 = torch.trapz(lorentzian_3[:,0:fermi], ergy[None,0:fermi])
        AvgVal_3 = torch.trapz(lorentzian_3[:,0:fermi] * ergy[None,0:fermi], ergy[None,0:fermi])
        energy_NA_3 = d_hyb_3 - 2*AvgVal_3
        dos_ads_3 = wdos_3/(deno_3**2+wdos_3**2)/np.pi
        dos_ads_3 = dos_ads_3/torch.trapz(dos_ads_3, ergy[None,:])[:,None]
        
        energy = (self.Esp
                  + (energy_NA_1 + 2*(na_1+f)*alpha_1*beta_1*vad2)
                  + (energy_NA_2 + 2*(na_2+f)*alpha_2*beta_2*vad2) * 2
                  + (energy_NA_3 + 2*(na_3+f)*alpha_3*beta_3*vad2))
        
        ans = torch.cat(((energy_DFT-energy).view(-1, 1),
                         (d_cen_DFT-d_cen).view(-1, 1),
                         (width_DFT-width).view(-1, 1),
                         self.lamb*(dos_ads_1_DFT-dos_ads_1),
                         self.lamb*(dos_ads_2_DFT-dos_ads_2),
                         self.lamb*(dos_ads_3_DFT-dos_ads_3)),1)
        
        ans = ans.view(len(ans),1,-1)
        
        return ans, namodel_in
    
    def gcnn(self, gcnnmodel_in, **kwargs):
        # Do nothing
        return gcnnmodel_in, gcnnmodel_in