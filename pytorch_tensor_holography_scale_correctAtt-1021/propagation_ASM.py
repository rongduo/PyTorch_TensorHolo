
"""
This is the script that is used for the wave propagation using the angular spectrum method (ASM). Refer to 
Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company Publishers, 2005, for principle details.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import math
import torch
import numpy as np
import propagation_utils as utils
import torch.fft as tfft


def propagation_ASM(u_in, feature_size, wavelength, z, ampert_mask=None,
                    linear_conv=True, padtype='zero', return_H=False, 
                    precomped_H=None, return_H_exp=False, precomped_H_exp=None, 
                    dtype=torch.float32):
    """Propagates the input field using the angular spectrum method

    Inputs
    ------
    u_in: complex field of size (num_images, 1, height, width), complex tensor
        where the last two channels are real and imaginary values
    feature_size: (height, width) of individual holographic features in m
    wavelength: wavelength in m
    z: propagation distance
    apert_mask: the amsk for amperture or pupil
    linear_conv: if True, pad the input to obtain a linear convolution
    padtype: 'zero' to pad with zeros, 'median' to pad with median of u_in's
        amplitude
    return_H[_exp]: used for precomputing H or H_exp, ends the computation early
        and returns the desired variable
    precomped_H[_exp]: the precomputed value for H or H_exp
    dtype: torch dtype for computation at different precision

    Output
    ------
    tensor of size (num_images, 1, height, width, 2)
    """

    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)


    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))


        # frequency coordinates sampling 
        fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y, dtype=torch.float64, device=u_in.device)
        fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x, dtype=torch.float64, device=u_in.device)

        # momentum/reciprocal space
        FY, FX = torch.meshgrid(fy, fx)
        
        # transfer function 
        HH = 2 * math.pi * torch.sqrt(1 / torch.tensor(wavelength ** 2, device=u_in.device) - (FX ** 2 + FY ** 2))
        
        # cast the H matrix to torch.float32
        H_exp = HH.to(torch.float32)

    
    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # physical size of the field
        y, x = (dy * float(num_y), dx * float(num_x))


        # generate momentum/reciprocal space
        fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y, dtype=torch.float64, device=u_in.device)
        fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x, dtype=torch.float64, device=u_in.device)
        FY, FX = torch.meshgrid(fy, fx)


    if return_H_exp:
        return H_exp        

    if precomped_H is None:
        if isinstance(z, float):
            # multiply by distance
            H_exp = H_exp * z
            
            # band-limited ASM - Matsushima et al. (2009)
            fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
            fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength

            # insert two new dimensions
            FX = FX[np.newaxis, np.newaxis, ...]
            FY = FY[np.newaxis, np.newaxis, ...]

        else:
            #multiply with multiple depth
            H_exp = torch.mul(H_exp.unsqueeze(1), torch.tensor(z, device=u_in.device))  # [B, #dist, #C, H, W]

            # band-limited ASM - Matsushima et al. (2009)
            fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength[:, np.newaxis, :, :, :]  # [B, #dist, #C, 1, 1]
            fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength[:, np.newaxis, :, :, :]  # [B, #dist, #C, 1, 1]
            
            FX = FX[np.newaxis, np.newaxis, np.newaxis, ...].expand(z.shape[0], z.shape[1], wavelength.shape[2], num_y, num_x)
            FY = FY[np.newaxis, np.newaxis, np.newaxis, ...].expand(z.shape[0], z.shape[1], wavelength.shape[2], num_y, num_x)
        
        # get the band-limited filter
        H_filter = ((torch.abs(FX) < torch.tensor(fx_max).to(FX)) & (torch.abs(FY) < torch.tensor(fy_max).to(FY))).float().to(u_in.device)

        # compute the H matrix
        H = H_filter * torch.exp(1j * H_exp)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H:
        return H
    
    # angular spectrum
    U1 = tfft.fftshift(tfft.fftn(u_in, dim=(-2, -1), norm='ortho'), (-2, -1))
   
    # convolution of the system
    if isinstance(z, float):
        U2 = U1 * H 
    else:
        U2 = U1.unsqueeze(1) * H

    # Fourier transform of the convolution to the observation plane
    u_out = tfft.ifftn(tfft.ifftshift(U2, (-2, -1)), dim=(-2, -1), norm='ortho')

    if linear_conv:
        u_out = utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)

    return u_out


    '''
    #print('Here the shape of H is ', H.shape)
    # angular spectrum
    U1 = torch.fft(utils.ifftshift(u_in), 2, True)  #[B, C, H, W]
    
    if isinstance(z, float):
        # convolution of the system
        U2 = utils.mul_complex(H, U1)
    else:
        U2 = utils.mul_complex(H, U1.unsqueeze(1))

    # Fourier transform of the convolution to the observation plane
    u_out = utils.fftshift(torch.ifft(U2, 2, True))
    

    if linear_conv:
        # if len(u_out.shape) == 6:
        #     u_out_tmp = torch.flatten(u_out, 0,1)
        #     result = utils.crop_image(u_out_tmp, input_resolution)
        #     result = result.view(*u_out.shape[:3], *input_resolution, 2)
        # else:
        result = utils.crop_image(u_out, input_resolution)
        return result
    else:
        return u_out
    '''
