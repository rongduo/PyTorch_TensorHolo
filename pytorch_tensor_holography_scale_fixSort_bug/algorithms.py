
"""
This is the algorithm script used for the representative iterative CGH implementations, i.e., GS and SGD.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

import utils.utils as utils
from propagation_ASM import *


# 1. GS
def gerchberg_saxton(init_phase, target_amp, num_iters, prop_dist, wavelength, feature_size=6.4e-6,
                     phase_path=None, prop_model='ASM', propagator=None,
                     writer=None, dtype=torch.float32, precomputed_H_f=None, precomputed_H_b=None):
    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator

    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the GS.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param phase_path: path to save the results.
    :param prop_model: string indicating the light transport model, default 'ASM'. ex) 'ASM', 'fresnel', 'model'
    :param propagator: predefined function or model instance for the propagation.
    :param writer: tensorboard writer
    :param dtype: torch datatype for computation at different precision, default torch.float32.
    :param precomputed_H_f: pre-computed kernel for forward prop (SLM to image)
    :param precomputed_H_b: pre-computed kernel for backward propagation (image to SLM)

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """

    # initial guess; random phase
    slm_field = torch.stack((torch.ones_like(init_phase), init_phase), -1)

    # run the GS algorithm
    for k in range(num_iters):
        # SLM plane to image plane
        recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
                                            prop_model, dtype, precomputed_H_f)

        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        if k > 0 and k % 10 == 0:
            utils.write_gs_summary(slm_field, recon_field, target_amp, k, writer, prefix='test')

        # replace amplitude at the image plane
        recon_field = utils.replace_amplitude(recon_field, target_amp)

        # image plane to SLM plane
        slm_field = utils.propagate_field(recon_field, propagator, -prop_dist, wavelength, feature_size,
                                          prop_model, dtype, precomputed_H_b)

        # amplitude constraint at the SLM plane
        slm_field = utils.replace_amplitude(slm_field, torch.ones_like(target_amp))

    # return phases
    _, final_slm_phase = utils.rect_to_polar(slm_field[..., 0], slm_field[..., 1])

    return final_slm_phase


# 2. SGD
def stochastic_gradient_descent(init_phase, target_amp, num_iters, prop_dist, wavelength, feature_size,
                                roi_res=None, phase_path=None, prop_model='ASM', propagator=None,
                                loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0,
                                writer=None, dtype=torch.float32, precomputed_H=None):

    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------
    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param phase_path: a string, for saving intermediate phases
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: pre-computed kernel shape of (1,1,2H,2W,2) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """

    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)

    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)

    # optimization variables and adam optimizer
    optvars = [{'params': slm_phase}]
    if lr_s > 0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    # crop target roi
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)

    # run the iterative algorithm
    for k in range(num_iters):
        optimizer.zero_grad()
        # forward propagation from the SLM plane to the target plane
        real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.stack((real, imag), -1)
        recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
                                            prop_model, dtype, precomputed_H)

        # get amplitude
        recon_amp, recon_phase = utils.rect_to_polar(recon_field[..., 0], recon_field[..., 1])

        # crop roi
        out_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)

        # calculate loss and backprop
        lossValue = loss(s * out_amp, target_amp)
        lossValue.backward()
        optimizer.step()

        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        with torch.no_grad():
            if k % 10 == 0:
                utils.write_sgd_summary(slm_phase, out_amp, target_amp, k,
                                        writer=writer, path=phase_path, s=s, prefix='test')

    return slm_phase




def stochastic_gradient_descent_joint(init_phase, target_amp, num_iters, feature_size,  wave_list=None, dist_list=None,
                                roi_res=None, phase_path=None, prop_model='ASM', propagator=None,
                                loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0,
                                writer=None, dtype=torch.float32, preH_list = [None, None, None]):

    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------
    :param init_phase: a tensor, in the shape of (1,3,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,3,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param phase_path: a string, for saving intermediate phases
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: pre-computed kernel shape of (1,1,2H,2W,2) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """
    if dist_list is None:
        dist_list = (20 * 0.01, 20 * 0.01, 20 * 0.01)# propagation distance from SLM plane to target plane
    if wave_list is None:
        wave_list = (638 * 1e-9, 520 * 1e-9, 450 * 1e-9)

    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)

    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)

    # optimization variables and adam optimizer
    optvars = [{'params': slm_phase}]
    if lr_s > 0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    # crop target roi
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)
    print(target_amp.shape)

    # run the iterative algorithm
    print(optimizer.state_dict)
    lossValue = None
    for k in range(num_iters):
        optimizer.zero_grad()
        amp_list = []
        for c in range(init_phase.size(1)):
            # forward propagation from the SLM plane to the target plane
            cur_slm_phase = slm_phase[:,c,:,:].unsqueeze(1)
            prop_dist = dist_list[c]
            wavelength = wave_list[c]
            real, imag = utils.polar_to_rect(torch.ones_like(cur_slm_phase), cur_slm_phase)
            slm_field = torch.stack((real, imag), -1)
            recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
                                                prop_model, dtype, preH_list[c])

            # get amplitude
            recon_amp, recon_phase = utils.rect_to_polar(recon_field[..., 0], recon_field[..., 1])

            # crop roi
            out_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)

            # calculate loss and backprop
            amp_list.append(out_amp)
            
        final_out_amp = torch.cat(amp_list, dim=1)
        lossValue = loss(s * final_out_amp, target_amp)
        lossValue.backward()
        print(torch.mean(init_phase.grad, dim=(2,3)))
        optimizer.step()

        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        with torch.no_grad():
            if k % 10 == 0:
                utils.write_sgd_summary(slm_phase, final_out_amp, target_amp, k,
                                        writer=writer, path=phase_path, s=s, prefix='test')

    return slm_phase


# 3. DPAC
def double_phase_amplitude_coding(target_phase, target_amp, prop_dist, wavelength, feature_size,
                                  prop_model='ASM', propagator=None,
                                  dtype=torch.float32, precomputed_H=None):
    """
    Use a single propagation and converts amplitude and phase to double phase coding

    Input
    -----
    :param target_phase: The phase at the target image plane
    :param target_amp: A tensor, (B,C,H,W,A), the amplitude at the target image plane.
    :param prop_dist: propagation distance, in m.
    :param wavelength: wavelength, in m.
    :param feature_size: The SLM pixel pitch, in meters.
    :param prop_model: The light propagation model to use for prop from target plane to slm plane
    :param propagator: propagation_ASM
    :param dtype: torch datatype for computation at different precision.
    :param precomputed_H: pre-computed kernel - to make it faster over multiple iteration/images - calculate it once

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W,1)
    """
    real, imag = utils.polar_to_rect(target_amp, target_phase)
    target_field = torch.stack((real, imag), -1)

    slm_field = utils.propagate_field(target_field, propagator, prop_dist, wavelength, feature_size,
                                      prop_model, dtype, precomputed_H)

    slm_phase = double_phase(slm_field, three_pi=False, mean_adjust=True)

    return slm_phase


def double_phase(field, three_pi=True, mean_adjust=True):
    """Converts a complex field to double phase coding

    field: tensor with dims [..., height, width, 2], where the last dim is the
        real and imaginary components, stacked
    three_pi, mean_adjust: see double_phase_amp_phase
    """
    amplitudes, phases = utils.rect_to_polar(field[..., 0], field[..., 1])

    return double_phase_amp_phase(amplitudes, phases, three_pi, mean_adjust)


def double_phase_amp_phase(amplitudes, phases, three_pi=True, mean_adjust=True):
    """converts amplitude and phase to double phase coding

    amplitudes:  per-pixel amplitudes of the complex field
    phases:  per-pixel phases of the complex field
    three_pi:  if True, outputs values in a 3pi range, instead of 2pi
    mean_adjust:  if True, centers the phases in the range of interest
    """

    # normalize
    amplitudes = amplitudes / amplitudes.max()

    phases_a = phases - torch.acos(amplitudes)
    phases_b = phases + torch.acos(amplitudes)

    phases_out = phases_a
    phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
    phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

    if three_pi:
        max_phase = 3 * math.pi
    else:
        max_phase = 2 * math.pi

    if mean_adjust:
        phases_out -= phases_out.mean()

    return (phases_out + max_phase / 2) % max_phase - max_phase / 2
