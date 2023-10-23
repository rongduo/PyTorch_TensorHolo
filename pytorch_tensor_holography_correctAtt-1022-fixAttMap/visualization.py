import numpy as np
import random
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import copy
from utils_yujie import batch_fftshift2d, batch_ifftshift2d, exponent_complex2d, complex_mul2d
import torch.nn.functional as F
import math
import propagation_utils as utils 
from propagation_ASM import propagation_ASM


def bgr2rgb(input_img):
    return input_img[:,:,::-1]

def vis_holo(input_holo, d0=1000):
    h = 0.000532
    pix = 0.00465
    pi = 3.141592653

    N = min(input_holo.shape[1], input_holo.shape[2])
    L = pix * N
    input_holo = input_holo * 255.
    f = input_holo[0,...].detach().numpy()
    n = np.linspace(1, N, N)

    x = -L/2 + L / N * (n-1)
    y = x
    [yy, xx] = np.meshgrid(y, x)
    k = 2 * pi / h
    t = np.zeros((N, N), dtype=complex) + (0+1j)
    Fresnel = np.exp(0 + t * k / 2 / d0 * (np.power(xx,2) + np.power(yy, 2)))
    f2 = f * Fresnel;
    Uf = np.fft.fft2(f2);
    Uf = np.fft.fftshift(Uf)
    L0 = h * d0 * N / L

    n = np.linspace(1, N, N)
    x = -L0 /2 + L0 / N * (n-1)
    y = x
    [yy, xx] = np.meshgrid(y, x)
    complex_unit = 0+1j
    phase = np.exp(complex_unit * k * d0) / (complex_unit * h * d0) * np.exp(t * k /2 / d0 * (np.power(xx, 2) + np.power(yy, 2)))
    Uf = Uf * phase
    Uf = np.abs(Uf)
    Gmax = np.max(Uf) / 10.0
    Gmin = np.min(Uf) # because the self-inference are has strongest intensity in the original image
    Uf = np.clip(Uf, Gmin, Gmax)
    Uf = (Uf - np.min(Uf)) / (np.max(Uf) - np.min(Uf))
    return Uf

def vis_holo_torch(input_holo):
    h = 0.000532
    pix = 0.00465
    d0 = 1000
    pi = 3.141592653

    N = min(input_holo.shape[1], input_holo.shape[2])
    L = pix * N
    input_holo = input_holo * 255.
    f = input_holo[0,...]
    n = torch.linspace(1, N, N)

    x = -L/2 + L / N * (n-1)
    y = x
    [yy, xx] = torch.meshgrid(y, x)
    k = 2 * pi / h
    xx = xx.cuda()
    yy = yy.cuda()
    #t = torch.zeros((N, N), dtype=torch.cfloat) + (0+1j)
    Fresnel = torch.stack([torch.cos(k / 2 / d0 * (torch.pow(xx,2) + torch.pow(yy, 2))), torch.sin(k / 2 / d0 * (torch.pow(xx,2) + torch.pow(yy, 2)))], dim=-1)
    Fresnel = Fresnel.cuda().float()
    f = f.unsqueeze(len(f.shape))
    f2 = f * Fresnel;
    print('f2 shape is ', f2.shape)
    Uf = torch.fft(f2, 2);
    Uf = Uf.unsqueeze(0)
    Uf = batch_fftshift2d(Uf)
    
    L0 = h * d0 * N / L
    n = torch.linspace(1, N, N)
    x = -L0 /2 + L0 / N * (n-1)
    y = x
    [yy, xx] = torch.meshgrid(y, x)
    yy = yy.cuda()
    xx = xx.cuda()
    complex_unit = torch.zeros(1,1, dtype=torch.cfloat) + (0+1j)
    complex_unit = complex_unit.cuda()
    phase_part1 = torch.stack([torch.sin(torch.Tensor([k * d0])) / (h * d0) , 0.0-torch.sin(torch.Tensor([k * d0])) / (h * d0)], dim=-1)
    phase_part1 = phase_part1.cuda()
    phase_part1 = phase_part1.unsqueeze(0)

    coor_part = torch.pow(xx, 2) + torch.pow(yy, 2)
    phase_part2 = torch.stack([torch.cos(k /2 / d0 * coor_part ), torch.sin(k /2 / d0 * coor_part)], dim=-1)
    phase = phase_part1 * phase_part2

   # phase = torch.exp(complex_unit * k * d0) / (complex_unit * h * d0) * torch.exp(t * k /2 / d0 * (torch.pow(xx, 2) + torch.pow(yy, 2)))
    result = Uf.new(Uf.size()).fill_(0.0)
    result[0,:,:,0] = Uf[0,:,:,0] * phase[:,:,0] -  Uf[0,:,:,1] * phase[:,:,1]
    result[0,:,:,1] = Uf[0,:,:,1] * phase[:,:,0] +  Uf[0,:,:,0] * phase[:,:,1]
    
    #Uf = Uf * phase
    Uf = torch.norm(result, p=2, dim=-1)
    #Uf = torch.abs(Uf)
    Gmax = torch.max(Uf).item() / 10.0
    Gmin = torch.min(Uf).item() # because the self-inference are has strongest intensity in the original image
    Uf = torch.clamp(Uf, Gmin, Gmax)
    Uf = (Uf - Gmin) / (Gmax - Gmin)
    return Uf


def i_fft(U, M, N, wave_length, z, xx0, yy0, xx, yy):
    k  = 2 * np.pi / wave_length
    G_left = torch.cat([torch.Tensor([0]), torch.Tensor([wave_length * z])], 0)
    G_middle = torch.cat([torch.cos(-k * z), torch.sin(-k*z)], 0)
    G_first = (G_left[0] * G_middle[0] + G_left[1] * G_middle[1] ) / torch.sum(torch.pow(G_middle, 2),-1)
    G_second = (G_left[1] * G_middle[0] - G_left[0] * G_middle[1] ) / torch.sum(torch.pow(G_middle, 2),-1)
    G_left = torch.stack([G_first, G_second], dim=0)
    
    G_right = torch.stack([torch.cos(-(torch.pow(xx,2) + torch.pow(yy,2))), torch.sin(-(torch.pow(xx,2)+torch.pow(yy,2)))], -1)
    G_real = G_right[:,:,0] * G_left[0] - G_right[:,:,1] * G_left[1]
    G_imag = G_right[:,:,0]* G_left[1] + G_right[:,:,1] * G_left[0]
    G = torch.stack([G_real, G_imag], dim=-1)
    UG_real = U[:,:,0] * G[:,:,0] - U[:,:,1] * G[:,:,1]
    UG_imag = U[:,:,1] * G[:,:,0] + U[:,:,0] * G[:,:,1]
    field = torch.stack([UG_real, UG_imag], -1)
    fft_result = torch.ifft(field, 2) 
    phase_shift_angle = (-1) * k * 2 / z * (torch.pow(xx0,2)+ torch.pow(yy0,2))
    phase_shift_real = torch.cos(phase_shift_angle)
    phase_shift_imag = torch.sin(phase_shift_angle)
    recon_real = fft_result[:,:,0] * phase_shift_real - fft_result[:,:,1] * phase_shift_imag
    recon_imag = fft_result[:,:,1] * phase_shift_real + fft_result[:,:,0] * phase_shift_imag
    U0 = torch.stack([recon_real, recon_imag], dim=-1)
    return U0


def recon_phaseHolo(pred_phase, target_depth):
    wave_length = 532 * 1e-9
    pixel_pitch = 2.4e-6
    res = 640
    x = np.linspace(-res//2-1, res//2, res) * pixel_pitch
    y = np.linspace(-res//2-1, res//2, res) * pixel_pitch
    xx,yy = np.meshgrid(x,y)
    xx = torch.from_numpy(xx).to(pred_phase.device)
    yy = torch.from_numpy(yy).to(pred_phase.device)
    delta_ovserve = wave_length *  target_depth / pixel_pitch / res 

    xx0 = xx / pixel_pitch * delta_ovserve
    yy0 = yy / pixel_pitch * delta_ovserve

    pred_phase = pred_phase[0,...] * np.pi * 3
    init_field = torch.stack([torch.cos(pred_phase), torch.sin(pred_phase)], dim=-1)
    recon = i_fft(init_field, init_field.shape[0], init_field.shape[1], wave_length, target_depth, xx0, yy0, xx, yy)
    #recon = torch.abs(recon)
    recon = torch.norm(recon, p=2, dim=-1)
    max_value = torch.max(recon).item() / 5.0
    min_value = torch.min(recon).item()
    recon = torch.clamp(recon, min_value, max_value)
    recon = (recon - min_value) / (max_value - min_value)
    recon = recon.unsqueeze(0).float()
    return recon

def recon_wirtinger(phase_map, prop_dist, hlimit=None, scale=None, is_test=False):
    '''
    input phase map size:  1 x Ny x Nx
    prop_dist size: 1,
    hlimit: None or 1 x Ny x Nx
    '''
    double_sampling = True
    alpha = 0.0429 
    pad_factor = 2
    

    phase_map = phase_map[0,...] * np.pi * 2
    #print(phase_map.shape, phase_map.max(),phase_map.min())
    Nx = phase_map.size(1)
    Ny = phase_map.size(0)
    wave_length = [6.3610 * 1e-4]
    pixel_pitch = 0.0064
    padding = False

    if double_sampling:
        pad_h = Ny // 2
        pad_w = Nx // 2 
        phase_map = F.pad(phase_map, (pad_h, pad_h, pad_w, pad_w), 'constant', 0)
        #input_map = torch.stack([phase_map.new(phase_map.size()).fill_(0.0), phase_map], dim=2)
    init_field = torch.stack([torch.cos(phase_map), torch.sin(phase_map)],dim=2)
        
    if double_sampling:
        field_width = 2 * Nx * pixel_pitch
        field_height = 2 * Ny * pixel_pitch
        X_tensor = torch.linspace(-Nx + 0.5, Nx - 1 + 0.5, 2 * Nx)
        Y_tensor = torch.linspace(-Ny + 0.5, Ny - 1 + 0.5, 2 * Ny)
    else:
        field_width = Nx * pixel_pitch
        field_height = Ny * pixel_pitch
        if Ny % 2 == 0:       
            Y_tensor = torch.linspace(-Ny // 2 + 0.5, Ny // 2 - 1 + 0.5,   Ny)
        else:
            Y_tensor = torch.linspace(-Ny // 2 + 0.5, Ny // 2 + 0.5 , Ny)

        if Nx % 2 != 0:
            X_tensor = torch.linspace(-Nx // 2 + 0.5, Nx // 2 + 0.5,  Nx)
        else:
            X_tensor = torch.linspace(-Nx // 2 + 0.5, Nx // 2 - 1 + 0.5,  Nx)
    X_tensor = X_tensor.to(phase_map.device)
    Y_tensor = Y_tensor.to(phase_map.device)
    [Y,X] = torch.meshgrid(Y_tensor, X_tensor)

    if hlimit == None:
        Hlimit_list = []
        for w in range(len(wave_length)):
            H = 2 * np.pi * prop_dist * torch.sqrt( 1/ (wave_length[w] * wave_length[w]) 
                                                        - torch.pow(1 / field_width * X, 2) - torch.pow( 1/ field_height * Y, 2))
            #H = exponent_complex2d(torch.stack([H.new(H.size()).fill_(0.0), H], dim=2))
            H = torch.stack([torch.cos(H), torch.sin(H)], dim=2)
            u_limit_x = 1 / ( math.sqrt((2 * prop_dist[0] * (1 / field_width)) * (2 * prop_dist * (1 / field_width)) + 1) * wave_length[w])
            u_limit_y = 1 / ( math.sqrt((2 * prop_dist[0] * (1 / field_height)) * (2 * prop_dist * (1 / field_height)) +1) * wave_length[w])
            N_limit_x = math.floor(u_limit_x / (1 / field_width))
            N_limit_y = math.floor(u_limit_y / (1 / field_height))

            if double_sampling:
                U_LIMIT = phase_map.new(2 * Ny, 2 * Nx).fill_(0.0)
                U_LIMIT[max(Ny - N_limit_y, 0) : min(Ny + N_limit_y, 2 * Ny-1),  max(Nx - N_limit_x, 0) : min(Nx + N_limit_x, 2 * Nx-1)] = 1
            else:
                U_LIMIT = torch.zeros(Ny, Nx)
                U_LIMIT[max(Ny//2 - N_limit_y, 0) : min(Ny //2 + N_limit_y, Ny-1),  max(Nx // 2 - N_limit_x, 0) : min(Nx // 2 + N_limit_x, Nx-1)] = 1
            U_LIMIT = U_LIMIT.unsqueeze(2)
            Hlimit_list.append( H * U_LIMIT)
        H_LIMIT = torch.stack(Hlimit_list, dim=0)
    else:
        hlimit = hlimit.permute(1,2,0).contiguous()
        H_LIMIT = hlimit.unsqueeze(0)

    '''
    Adjusting the overall energy in the image plane 
    Iin is a mask in which the value  of the padding area is small and the  value of the center area is 1.
    '''
    Iin = phase_map.new(Ny, Nx).fill_(1.0)
    if padding:
        id1 = phase_map.new(Ny, Nx).fill_(0.0)
        id1[ math.floor(Ny * (2 / (pad_factor + 2))) : math.floor(Ny * ((pad_factor + 1) / (pad_factor + 2))) - 1,
                math.floor(Nx * (2 / (pad_factor + 2))) :  math.floor(Ny * ((pad_factor + 1) / (pad_factor + 2))) - 1] = 1
        Iin = Iin * id1 + (1 - id1) * Iin * alpha
    else:
        id1 = phase_map.new(Ny, Nx).fill_(0.0)
        id1[:Ny, :] = 1
        Iin = Iin * (1 - id1) + id1 * Iin 

    if double_sampling:
        Iin = F.pad(Iin, (pad_h, pad_h, pad_w, pad_w), 'constant', 0)

    ########################################################################################################
    #ASM reconstruction
    #######################################################################################################
    objectiveField = Iin.unsqueeze(2) * init_field
    objectiveField = objectiveField.unsqueeze(0)
    objectiveField_FFT = batch_fftshift2d(torch.fft(batch_ifftshift2d(objectiveField), 2))
    objectiveField_HLIMIT = complex_mul2d(objectiveField_FFT, H_LIMIT)
    image_recon = batch_fftshift2d(torch.ifft(batch_ifftshift2d(objectiveField_HLIMIT), 2))
    image = torch.pow(torch.norm(image_recon, p=2, dim=3), 2)   
    


    image = torch.clamp(image, 0,  image.max().item() / 5)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)

    if not is_test:
        return image
    else:
        y_select = torch.arange(Ny//2, Ny //2 * 3).to(phase_map.device)
        x_select = torch.arange(Nx//2, Nx //2 * 3).to(phase_map.device)
        crop_img_y = torch.index_select(image, dim=1, index=y_select)
        crop_img = torch.index_select(crop_img_y, dim=2, index=x_select)
        return crop_img
  
    #crop_img = image[:, Ny // 2 :  Ny  // 2 * 3, Nx // 2 :  Nx // 2 * 3]
    '''
    mask = torch.ones(Ny, Nx).to(phase_map.device).unsqueeze(0)
    mask = F.pad(mask, (pad_h, pad_h, pad_w, pad_w), 'constant', 0)
    image_mask = torch.mul(image, mask) 
    '''
    return image_resize



def recon_sgdHolo_specific(input_amp, input_phase, wavelength_list, prop_dist, feature_size):
    slm_res = input_phase.shape[-2:]
    propagator = propagation_ASM
    recon_amp = []

    for c in range(input_phase.shape[1]):
        cur_amp = input_amp[:, c, ...].reshape(input_phase.size(0),1,*slm_res)
        cur_phase = input_phase[:, c, ...].reshape(input_phase.size(0), 1, *slm_res)
        real, imag = utils.polar_to_rect(cur_amp, cur_phase)
        slm_field = torch.stack((real, imag), -1)
        recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength_list[c],
                                            feature_size, 'ASM', dtype=torch.float32)
        recon_amp_c, recon_phs_c = utils.rect_to_polar(recon_field[...,0], recon_field[...,1])
        recon_amp.append(recon_amp_c)
    final_recon_amp = torch.cat(recon_amp, dim=1)
    return final_recon_amp

