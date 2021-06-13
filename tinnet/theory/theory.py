'''
collection of chemisorption models.

newns_anderson:
'''

import torch
import numpy as np


class Chemisorption:

    def __init__(self, model_name, main_target, **kwargs):
        # Initialize the class
        if model_name == 'gcnn':
            self.model_num_input = 1
            self.target = main_target
            
        if model_name == 'newns_anderson_semi':
            self.model_num_input = 12
            
            cuda = torch.cuda.is_available()
            
            # h for Hilbert Transform    
            num_datapoints = 3001
            emin = -15
            emax = 15
            h = np.zeros(num_datapoints)
            if num_datapoints % 2 == 0:
                h[0] = h[num_datapoints // 2] = 1
                h[1:num_datapoints // 2] = 2
            else:
                h[0] = 1
                h[1:(num_datapoints+1) // 2] = 2
            
            # ergy for dos
            ergy = np.linspace(emin, emax, num_datapoints)
            
            # target(s)
            esp = kwargs['constant_1']
            
            try:
                root_lamb = kwargs['constant_2']
            except:
                root_lamb = 0.1
            
            vad2 = kwargs['constant_3']
            
            try:
                d_cen = kwargs['additional_traget_1']
            except:
                d_cen = np.zeros((len(main_target),1))
            
            try:
                half_width = kwargs['additional_traget_2']
            except:
                half_width = np.zeros((len(main_target),1))
            
            try:
                dos_ads_3sigma = kwargs['additional_traget_3']
            except:
                dos_ads_3sigma = np.zeros((len(main_target),num_datapoints))
            
            try:
                dos_ads_1pi = kwargs['additional_traget_4']
            except:
                dos_ads_1pi = np.zeros((len(main_target),num_datapoints))
            
            try:
                dos_ads_4sigma = kwargs['additional_traget_5']
            except:
                dos_ads_4sigma = np.zeros((len(main_target),num_datapoints))
            
            main_target = np.reshape(main_target,(-1,1))
            d_cen = np.reshape(d_cen,(-1,1))
            half_width = np.reshape(half_width,(-1,1))
            
            self.target = np.hstack((main_target,
                                     d_cen,
                                     half_width,
                                     root_lamb*dos_ads_3sigma,
                                     root_lamb*dos_ads_1pi,
                                     root_lamb*dos_ads_4sigma))
            
            # Initialize the class
            h = torch.FloatTensor(h)
            ergy = torch.FloatTensor(ergy)
            vad2 = torch.from_numpy(vad2).type(torch.FloatTensor)
            
            if cuda:
                h = h.cuda()
                ergy = ergy.cuda()
                vad2 = vad2.cuda()
            
            self.h = h
            self.ergy = ergy
            self.esp = esp
            self.root_lamb = root_lamb
            self.vad2 = vad2
            
        if model_name == 'user_defined':
            self.model_num_input = 1
            self.target = main_target
        
    def newns_anderson_semi(self, namodel_in, dos_source, target, **kwargs):
        
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
        model_d_cen = namodel_in[:,10]
        model_half_width = torch.nn.functional.softplus(namodel_in[:,11])
        
        idx = kwargs['batch_cif_ids']
        
        vad2 = self.vad2[idx]
        dft_ead = target[:,0,0]
        dft_d_cen = target[:,0,1]
        dft_half_width = target[:,0,2]
        ergy = self.ergy
        
        self.fermi = np.argsort(abs(ergy.detach().cpu().numpy()))[0] + 1
        
        # Semi-ellipse
        if dos_source == 'dft':
            dos_d = 1-((ergy[None,:]-dft_d_cen[:,None])
                       / dft_half_width[:,None])**2
            dos_d = abs(dos_d)**0.5
            dos_d *= (abs(ergy[None,:]-dft_d_cen[:,None]) 
                      < dft_half_width[:,None])
            dos_d += (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
        else:
            dos_d = 1-((ergy[None,:]-model_d_cen[:,None])
                       / model_half_width[:,None])**2
            dos_d = abs(dos_d)**0.5
            dos_d *= (abs(ergy[None,:]-model_d_cen[:,None])
                      < model_half_width[:,None])
            dos_d += (torch.trapz(dos_d,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d = dos_d / torch.trapz(dos_d,ergy)[:,None]
        
        filling = torch.trapz(dos_d[:,0:self.fermi],ergy[0:self.fermi])
        
        na_1, energy_NA_1, model_dos_ads_3sigma \
            = Chemisorption.NA_Model(self, adse_1, beta_1, delta_1, dos_d,
                                     vad2)
        
        na_2, energy_NA_2, model_dos_ads_1pi \
            = Chemisorption.NA_Model(self, adse_2, beta_2, delta_2, dos_d,
                                     vad2)
        
        na_3, energy_NA_3, model_dos_ads_4sigma \
            = Chemisorption.NA_Model(self, adse_3, beta_3, delta_3, dos_d,
                                     vad2)
        
        ed_hybridization = energy_NA_1 + energy_NA_2 * 2 + energy_NA_3
        
        ed_repusion = (2*(na_1+filling)*alpha*beta_1*vad2
                       + 4*(na_2+filling)*alpha*beta_2*vad2
                       + 2*(na_3+filling)*alpha*beta_3*vad2)
        
        model_energy = self.esp + ed_hybridization + ed_repusion
        
        if torch.cuda.is_available():
            idx = torch.from_numpy(np.array(idx, dtype=np.float32)).cuda()
        else:
            idx = torch.from_numpy(np.array(idx, dtype=np.float32))
        
        parm = torch.stack((idx,
                            dft_ead,
                            dft_d_cen,
                            dft_half_width,
                            ed_hybridization,
                            ed_repusion,
                            model_energy,
                            model_d_cen,
                            model_half_width,
                            adse_1,
                            beta_1,
                            delta_1,
                            adse_2,
                            beta_2,
                            delta_2,
                            adse_3,
                            beta_3,
                            delta_3,
                            alpha,
                            filling,
                            vad2)).T
        
        ans = torch.cat((model_energy.view(-1, 1),
                         model_d_cen.view(-1, 1),
                         model_half_width.view(-1, 1),
                         self.root_lamb*model_dos_ads_3sigma,
                         self.root_lamb*model_dos_ads_1pi,
                         self.root_lamb*model_dos_ads_4sigma),1)
        
        ans = ans.view(len(ans),1,-1)
        
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

    def user_defined(self, user_defined_model_in, **kwargs):
        # Do something:
        return user_defined_model_in, user_defined_model_in
    