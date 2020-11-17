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
            self.model_num_input = 12
    
    def newns_anderson_semi(self, namodel_in, model, task , **kwargs):
        
        adse_1 = namodel_in[:,0]
        beta_1 = torch.nn.functional.softplus(namodel_in[:,1])
        delta_1 = torch.nn.functional.softplus(namodel_in[:,2])
        adse_2 = namodel_in[:,3]
        beta_2 = torch.nn.functional.softplus(namodel_in[:,4])
        delta_2 = torch.nn.functional.softplus(namodel_in[:,5])
        adse_3 = namodel_in[:,6]
        beta_3 = torch.nn.functional.softplus(namodel_in[:,7])
        delta_3 = torch.nn.functional.softplus(namodel_in[:,8])
        alpha = torch.sigmoid(namodel_in[:,9])
        d_cen = namodel_in[:,10]
        width = torch.nn.functional.softplus(namodel_in[:,11])
        
        idx = kwargs['batch_cif_ids']
        
        energy_DFT = self.energy[idx]
        vad2 = self.vad2[idx]
        
        if task == 'train':
            d_cen_DFT = self.d_cen[idx]
            width_DFT = self.width[idx]
            dos_ads_1_DFT = self.dos_ads_1[idx]
            dos_ads_2_DFT = self.dos_ads_2[idx]
            dos_ads_3_DFT = self.dos_ads_3[idx]
        
        ergy = self.ergy
        
        self.fermi = np.argsort(abs(ergy.detach().cpu().numpy()))[0] + 1
        
        # Semi-ellipse
        if model == 'dft':
            dos_d = 1-((ergy[None,:]-d_cen_DFT[:,None])/width_DFT[:,None])**2
            dos_d = abs(dos_d)**0.5
            dos_d *= (abs(ergy[None,:]-d_cen_DFT[:,None]) < width_DFT[:,None])
            dos_d += (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
        else:
            dos_d = 1-((ergy[None,:]-d_cen[:,None])/width[:,None])**2
            dos_d = abs(dos_d)**0.5
            dos_d *= (abs(ergy[None,:]-d_cen[:,None]) < width[:,None])
            dos_d += (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
        
        f = torch.trapz(dos_d[:,0:self.fermi],ergy[0:self.fermi])
        
        na_1, energy_NA_1, dos_ads_1 = Chemisorption.NA_Model(self, adse_1,
                                                              beta_1, delta_1,
                                                              dos_d, vad2)
        
        na_2, energy_NA_2, dos_ads_2 = Chemisorption.NA_Model(self, adse_2,
                                                              beta_2, delta_2,
                                                              dos_d, vad2)
        
        na_3, energy_NA_3, dos_ads_3 = Chemisorption.NA_Model(self, adse_3,
                                                              beta_3, delta_3,
                                                              dos_d, vad2)
        
        energy = (self.Esp
                  + (energy_NA_1 + 2*(na_1+f)*alpha*beta_1*vad2)
                  + (energy_NA_2 + 2*(na_2+f)*alpha*beta_2*vad2) * 2
                  + (energy_NA_3 + 2*(na_3+f)*alpha*beta_3*vad2))
        
        idx = torch.from_numpy(np.array(idx, dtype=np.float32)).cuda()
        
        parm = torch.stack((idx,
                            energy_DFT,
                            energy,
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
                            delta_3,
                            alpha)).T
        if task == 'train':
            ans = torch.cat(((energy_DFT-energy).view(-1, 1),
                             (d_cen_DFT-d_cen).view(-1, 1),
                             (width_DFT-width).view(-1, 1),
                             self.lamb*(dos_ads_1_DFT-dos_ads_1),
                             self.lamb*(dos_ads_2_DFT-dos_ads_2),
                             self.lamb*(dos_ads_3_DFT-dos_ads_3)),1)
            ans = ans.view(len(ans),1,-1)
        
        elif task == 'test':
            ans = energy.view(len(energy),-1)
        
        return ans, parm
    
    def NA_Model(self, adse, beta, delta, dos_d, vad2):
        h = self.h
        ergy = self.ergy
        eps = np.finfo(float).eps
        fermi = self.fermi
        
        wdos = np.pi * (beta[:,None]*vad2[:,None]*dos_d) + delta[:,None]
        wdos_ = np.pi * (0*vad2[:,None]*dos_d) + delta[:,None]
        
        # Hilbert transform
        af = torch.rfft(wdos,1,onesided=False)
        htwdos = torch.ifft(af*h[None,:,None],1)[:,:,1]
        deno = (ergy[None,:] - adse[:,None] - htwdos)
        deno = deno * (torch.abs(deno) > eps) \
               + eps * (torch.abs(deno) <= eps) * (deno >= 0) \
               - eps * (torch.abs(deno) <= eps) * (deno < 0)
        integrand = wdos / deno
        arctan = torch.atan(integrand)
        arctan = (arctan-np.pi)*(arctan > 0) + (arctan)*(arctan <= 0)
        d_hyb = 2 / np.pi * torch.trapz(arctan[:,0:fermi],ergy[None,0:fermi])
        
        lorentzian = (1/np.pi) * (delta[:,None]) \
                     / ((ergy[None,:] - adse[:,None])**2 + delta[:,None]**2)
        na = torch.trapz(lorentzian[:,0:fermi], ergy[None,0:fermi])
        
        deno_ = (ergy[None,:] - adse[:,None])
        deno_ = deno_ * (torch.abs(deno_) > eps) \
                + eps * (torch.abs(deno_) <= eps) * (deno_ >= 0) \
                - eps * (torch.abs(deno_) <= eps) * (deno_ < 0)
        integrand_ = wdos_ / deno_
        arctan_ = torch.atan(integrand_)
        arctan_ = (arctan_-np.pi)*(arctan_ > 0) + (arctan_)*(arctan_ <= 0)
        d_hyb_ = 2 / np.pi * torch.trapz(arctan_[:,0:fermi],ergy[None,0:fermi])
        
        energy_NA = d_hyb - d_hyb_
        
        dos_ads = wdos/(deno**2+wdos**2)/np.pi
        dos_ads = dos_ads/torch.trapz(dos_ads, ergy[None,:])[:,None]
        return na, energy_NA, dos_ads
    
    def gcnn(self, gcnnmodel_in, **kwargs):
        # Do nothing
        return gcnnmodel_in, gcnnmodel_in
