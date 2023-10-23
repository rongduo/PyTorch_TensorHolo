import os 
import numpy as np 
import torch 
import propagation_utils as utils 
from propagation_ASM import propagation_ASM
import time 


class holo_propagator(torch.nn.Module):
    def __init__(self, wavelength, feature_size=(6.4e-6, 6.4e-6), precomped_H_exp=None):
        super(holo_propagator, self).__init__()
        self.precomped_H_exp = precomped_H_exp
        if isinstance(wavelength, list):
            self.wavelength = np.array(wavelength).reshape((1,len(wavelength),1,1))
        else:
            self.wavelength = np.array([wavelength]).reshape((1,1,1,1))
                    
        self.feature_size = feature_size
        self.propagator = propagation_ASM
    
    def forward(self, input_phase,  prop_dist, input_amp=None, return_H=False):
        #print('in propagator, feature_size is  ', self.feature_size)
        if isinstance(prop_dist, float):
            prop_dist = prop_dist 
        elif isinstance(prop_dist, torch.Tensor):
            prop_dist = prop_dist.cpu().numpy()
            prop_dist = prop_dist[:, :, np.newaxis, np.newaxis, np.newaxis]
            #prop_dist = prop_dist.cpu().numpy().reshape((prop_dist.shape[0], prop_dist.shape[1], 1, 1, 1)).astype(np.float32)
        elif isinstance(prop_dist, list):
            prop_dist = np.array(prop_dist).reshape(1, len(prop_dist), 1, 1, 1).astype(np.float32)
        else:
            raise NotImplementedError 

        
        # Create a complex wavefront field on the slm plane
        if input_amp is not None and input_phase is not None:
            # for a complex hologram, just construct the wavefront using the input amplitude and phase
            slm_field = input_amp * torch.exp(1j * input_phase)
        elif input_amp is not None:
            # for binary amplitude-type hologram, do fft first and use the upper half of the spectrum
            slm_field = torch.fft.ifftn(torch.fft.ifftshift(input_amp)).cuda() # FT
            H = slm_field.shape[-2]
            slm_field = slm_field[..., 0:int(H/2), ...]  
        elif input_phase is not None:
            # for a phase-only hologram, just assume the amplitude is uniform and construct the field
            slm_field = torch.exp(1j * input_phase)
            

        if self.precomped_H_exp is None:
            self.precomped_H_exp = self.propagator(slm_field,
                                         self.feature_size,
                                         self.wavelength,
                                         prop_dist,
                                         return_H_exp=True)
            self.precomped_H_exp = self.precomped_H_exp.to(input_phase).detach()
            self.precomped_H_exp.requires_grad = False

        if return_H:
            self.precomped_H = self.propagator(slm_field,
                                         self.feature_size,
                                         self.wavelength,
                                         prop_dist,
                                         precomped_H_exp = self.precomped_H_exp,
                                         return_H=True)
            #self.precomped_H = self.precomped_H.to(input_phase).detach()
            # In torch 1.10.0, we need to specify clearly in .to() so that it will only move the tensor to the gpu without changing its type to complex32.
            self.precomped_H = self.precomped_H.to(input_phase.device).detach() 

        output_field = self.propagator(u_in = slm_field, z = prop_dist, feature_size = self.feature_size, 
                                    wavelength = self.wavelength, dtype = torch.float32, 
                                    precomped_H_exp=self.precomped_H_exp)

        # obtain the amplitude and phase from the output wavefield
        recon_amp =  output_field.abs()
        recon_phs =  output_field.angle()
        return recon_amp, recon_phs     



if __name__ == '__main__':
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    feature_size = [8*um, 8*um]
    wavelength_list = [638*nm, 520*nm, 450*nm]
    propagator = holo_propagator(wavelength_list, feature_size)
    propagator_asm = propagation_ASM

    input_phase = 2 * (torch.randn((2, 3, 512, 512)) - 0.5) * np.pi 
    input_phase = input_phase.to('cuda:0')
    input_amp = torch.ones_like(input_phase).to(input_phase)
    
    #prop_dist = 3 * mm
    prop_dist = torch.tensor([1 * mm, 2 * mm, 3 * mm, -1 * mm, -3 * mm]).unsqueeze(0).repeat(2, 1)
    recon_amp, recon_phase = propagator(input_phase, prop_dist, input_amp, return_H=True)
    print(recon_amp.shape)


    '''
    using previous slow for-loop to reconstruct focal stack for multi-channel
    '''

    start = time.time()
    all_recon_amps = []
    all_recon_phs = []
    all_Hexp_list = []
    all_H_list = []

    for b in range(input_phase.shape[0]):
        cur_amp =  input_amp[b, ...]
        cur_phase = input_phase[b, ...]
        cur_dist_list = prop_dist[b, ...]

        #H_exp_c_list = []
        '''Just used for debugging whethere there are difference between the H_exp for two reconstruction ways'''
        # for c in range(input_phase.shape[1]):
        #     H_exp_d_list = []
        #     for d in range(prop_dist.shape[1]):
        #         if d == 0:
        #             cur_dist = prop_dist[b, d]
        #             cur_amp_c = cur_amp[c, ...].unsqueeze(0).unsqueeze(0)
        #             cur_phs_c = cur_phase[c, ...].unsqueeze(0).unsqueeze(0)
        #             slm_field = torch.stack(utils.polar_to_rect(cur_amp_c, cur_phs_c), dim=-1)
        #             cur_wave_length = wavelength_list[c]
                    
        #             cur_H_exp = propagator_asm(slm_field, feature_size, cur_wave_length, cur_dist, return_H_exp=True)
        #             H_exp_c_list.append(cur_H_exp)
        #             #H_exp_d_list.append(cur_H_exp)
        #         # H_exp_d = torch.cat(H_exp_d_list, dim=0)
        #         #H_exp_c_list.append(H_exp_d)
                
        # H_exp_c = torch.cat(H_exp_c_list, dim=1)
        # # all_Hexp_list.append(H_exp_c)


        H_c_list = []
        H_exp_c_list = []
        '''Just used for debugging whethere there are difference between the H for two reconstruction ways'''
        for c in range(input_phase.shape[1]):
            H_d_list = []
            H_exp_d_list = []
            for d in range(prop_dist.shape[1]):
                cur_dist = prop_dist[b, d]
                cur_amp_c = cur_amp[c, ...].unsqueeze(0).unsqueeze(0)
                cur_phs_c = cur_phase[c, ...].unsqueeze(0).unsqueeze(0)
                slm_field = torch.stack(utils.polar_to_rect(cur_amp_c, cur_phs_c), dim=-1)
                cur_wave_length = wavelength_list[c]
                cur_wave_length = np.array([cur_wave_length]).reshape((1,1,1,1))
                
                cur_H = propagator_asm(slm_field, feature_size, cur_wave_length, cur_dist.item(), return_H=True)
                H_d_list.append(cur_H)

                if d == 0:
                    cur_H_exp = propagator_asm(slm_field, feature_size, cur_wave_length, cur_dist.item(), return_H_exp=True)
                    H_exp_d_list.append(cur_H_exp)
                
            H_d = torch.cat(H_d_list, dim=0)
            H_c_list.append(H_d)

            H_exp_d = torch.cat(H_exp_d_list, dim=0)
            H_exp_c_list.append(H_exp_d)
        
        H_c = torch.cat(H_c_list, dim=1)
        all_H_list.append(H_c)

        H_exp_c = torch.cat(H_exp_c_list, dim=1)

                
        cur_recon_amp, cur_recon_phase = recon_focal_stack(cur_amp, cur_phase, cur_dist_list, feature_size, wavelength_list)
        all_recon_amps.append(cur_recon_amp)
        all_recon_phs.append(cur_recon_phase)

    H_exp_batch_for = H_exp_c 
    error_Hexp = (H_exp_batch_for - propagator.precomped_H_exp).abs().mean()
    print('error for H_exp is ', error_Hexp.item())

    H_batch_for = torch.stack(all_H_list, dim=0)
    error_H = (H_batch_for - propagator.precomped_H).abs().mean()
    print('error for H is ', error_H.item())
    print('the shape of H_batch_for and propagator.precomped_H is ', H_batch_for.shape, propagator.precomped_H.shape)

    #print('the shape of H_exp_batch_for is ', H_exp_batch_for.shape)
    recon_amp_for = torch.stack(all_recon_amps, dim=0)
    recon_phase_for = torch.stack(all_recon_phs, dim=0)

    

    end = time.time()
    elapsed = end - start 
    print('recon with for-loop takes {:.5f} seconds'.format(elapsed))
    print('recon_amp (for loop) .shape is ', recon_amp_for.shape)
    print('recon phs (for loop) .shape is ', recon_phase_for.shape)

    error_amp = (recon_amp_for - recon_amp).abs().mean()
    error_phs = (recon_phase_for - recon_phase).abs().mean()

    


    print('error_amp and error_phase are ', error_amp.item(), error_phs.item())
    


'''
Use torch float64
/home/yujie/yujie_codes/DefocusHolo_new/PSF_learning_test/propagation_ASM.py:93: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  H_exp = torch.tensor(HH, dtype=dtype, device=u_in.device)
Use torch float64
use torch tensor in H construction
/home/yujie/yujie_codes/DefocusHolo_new/PSF_learning_test/propagation_ASM.py:168: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  H_filter = ((torch.abs(FX) < torch.tensor(fx_max).to(FX)) & (torch.abs(FY) < torch.tensor(fy_max).to(FY))).float().to(u_in.device)
self.precomped_H_exp shape is  torch.Size([1, 3, 1024, 1024])
Use torch float64
use torch tensor in H construction
recon multi-channel together taking 0.02596 seconds
torch.Size([2, 5, 3, 1024, 1024])
use numpy array in H construction
Use numpy float 64
use numpy array in H construction
Use numpy float 64
use numpy array in H construction
Use numpy float 64
use numpy array in H construction
Use numpy float 64

error for H_exp is  0.0
error for H is  0.0
the shape of H_batch_for and propagator.precomped_H is  torch.Size([2, 5, 3, 1024, 1024, 2]) torch.Size([2, 5, 3, 1024, 1024, 2])
recon with for-loop takes 11.27856 seconds
recon_amp (for loop) .shape is  torch.Size([2, 5, 3, 1024, 1024])
recon phs (for loop) .shape is  torch.Size([2, 5, 3, 1024, 1024])
error_amp and error_phase are  0.0 0.0
'''