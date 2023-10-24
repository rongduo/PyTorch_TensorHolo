from tqdm import tqdm
import argparse
from dataset import get_dataloader
from common import get_config
from agent import get_agent
from utils_yujie import ensure_dir
import cv2
import imageio
import numpy as np
import os
import torch
import random
from visualization import vis_holo, recon_wirtinger
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from metrics import ssim_matlab, calc_psnr
import json
import math 
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io as sio
from propagation_utils import crop_image 
from operator import itemgetter
import glob

def scale(x, label, weight=None):  ##### x.size() should be NxCxHxW
    #alpha = torch.cuda.FloatTensor(x.size())
    alpha = x.new(x.size())

    denominator = torch.mul(x, label)
    numerator = torch.mul(x, x)

    alpha_vector = torch.div(denominator.sum(-1).sum(-1),numerator.sum(-1).sum(-1))
    #print(alpha_vector)
    alpha_vector[alpha_vector != alpha_vector] = 0  #### changes nan to 0
    alpha_vector = torch.clamp(alpha_vector, min=0.1, max=10.0)

    for i in range(x.size(0)):
        for j in range(x.size(1)):
            alpha[i][j].fill_(alpha_vector[i][j])

    x = torch.mul(torch.autograd.Variable(alpha, requires_grad=False), x)
    return x

def save_img(img_array, file_name):
    if img_array.shape[0] == 3:
        img_array = img_array.transpose((1, 2, 0))
    img = (img_array * 255).astype(np.uint8)
    if img.shape[-1] ==3:
        img = img[..., ::-1]
    sio.imsave(file_name, img)

def save_fstack_video(fstack, file_name):
    fstack = fstack[..., ::-1]
    fstack = (fstack * 255).astype(np.uint8)
    imageio.mimwrite(file_name, fstack, fps=15, quality=8)

def cal_fstack_error(pred_stack, tgt_stack):
    psnr_list = []
    ssim_list = []
    psnr_scale_list = []
    ssim_scale_list = []
    all_imgs = []

    for m in range(pred_stack.shape[0]):
        cur_img_pred = pred_stack[m, ...].unsqueeze(0) 
        cur_img_tgt = tgt_stack[m,...].unsqueeze(0)

        # clip both the pred and tgt image to be within the range [0, 1]
        cur_img_pred = torch.clamp(cur_img_pred, 0, 1)
        cur_img_tgt = torch.clamp(cur_img_tgt, 0, 1)

        # scale the pred focal image to make it within the same range as the target 
        cur_img_scale = scale(cur_img_pred, cur_img_tgt)

        # transform them to numpy arrays
        cur_img_pred = cur_img_pred.squeeze().cpu().numpy()
        cur_img_tgt = cur_img_tgt.squeeze().cpu().numpy()
        cur_img_scale = cur_img_scale.squeeze().cpu().numpy()

        # move the channel dimesion to the last dimension
        if cur_img_pred.shape[0] == 3:
            cur_img_pred = cur_img_pred.transpose((1,2,0))
            cur_img_tgt = cur_img_tgt.transpose((1,2,0))
            cur_img_scale = cur_img_scale.transpose((1,2,0))

        # calculate psnr and ssim (before and after scaling)
        cur_psnr = psnr(cur_img_tgt, cur_img_pred)
        cur_ssim = ssim(cur_img_tgt, cur_img_pred, multichannel=True)
        cur_psnr_scale = psnr(cur_img_tgt, cur_img_scale)
        cur_ssim_scale = ssim(cur_img_tgt, cur_img_scale, multichannel=True)

        # append the value into the lists
        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)
        psnr_scale_list.append(cur_psnr_scale)
        ssim_scale_list.append(cur_ssim_scale)
        all_imgs.append(np.concatenate([cur_img_pred, cur_img_tgt], axis=-2))

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_psnr_scale = sum(psnr_scale_list) / len(psnr_scale_list)
    avg_ssim_scale = sum(ssim_scale_list) / len(ssim_scale_list)
    cat_stack = np.stack(all_imgs, axis=0)
    #print('cat_stack.shape is ', cat_stack.shape)

    return psnr_list, ssim_list, psnr_scale_list, ssim_scale_list, avg_psnr, avg_ssim, avg_psnr_scale, avg_ssim_scale, cat_stack


def read_img(filename, is_depth=False):
    img = cv2.imread(filename)
    img = img.astype(np.float32) / 255.0
    if is_depth:
        if len(img.shape) == 3:
            img = img[..., 0]
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    return img

def recon_largeRes_focalStack(input_phs, input_amp, recon_dists, propagator):
    all_imgs = []
    for m in range(recon_dists.shape[1]):
        recon_dist_m = torch.tensor([recon_dists[:, m].item()]).expand(1, 1)
        recon_dist_m = recon_dist_m.to(input_phs.device)
        pred_amp_m, pred_phase_m = propagator(input_phs, recon_dist_m * 1e-3, input_amp)
        all_imgs.append(pred_amp_m.squeeze())
    focal_stack = torch.stack(all_imgs, dim=0)
    print('focal_stack.shape is ', focal_stack.shape)
    return focal_stack


def test(rgb_path, depth_path, tr_agent, save_dir, postifx=''):
    count = 0

    rgb_tensor = read_img(rgb_path).cuda()
    depth_tensor = read_img(depth_path, is_depth=True).cuda()
    net_input = torch.cat([rgb_tensor, depth_tensor], dim=1) - 0.5

    output = tr_agent.net(net_input)
    out_amp = output[:,:3,...] * 0.5 + 0.5 # amplitude
    out_phase = output[:,3:,...] * np.pi # predicted phase
    recon_dist = torch.from_numpy(np.linspace(-3, 3, 30)).unsqueeze(0)
    pred_amp_stack = recon_largeRes_focalStack(out_phase, out_amp, recon_dist, tr_agent.propagator)

    input_img, input_depth = rgb_tensor, depth_tensor
    input_img = input_img.squeeze().cpu().numpy()
    input_depth = input_depth.squeeze().cpu().numpy()
    batch_size = 1

    for j in range(batch_size):
        pred_amp = out_amp[j, ...].squeeze().detach().cpu().numpy()
        pred_phs = out_phase[j, ...].squeeze().detach().cpu().numpy()
        
        if len(pred_amp.shape) == 3 and pred_amp.shape[0] == 3:
            pred_amp = pred_amp.transpose((1,2,0))
            pred_phs = pred_phs.transpose((1,2,0))
      
        pred_amp = np.clip(pred_amp, 0.0, 1.0)
        index = rgb_path.split('/')[-3]

        # save the produced images or videos
        pred_phs = (pred_phs + np.pi ) / (2 * np.pi) 
        
        holo_amp_name = os.path.join(save_dir, index + '-holo-amp-pred.png')
        holo_phs_name = os.path.join(save_dir, index  + '-holo-phase-pred.png')

        save_img(pred_amp, holo_amp_name)
        save_img(pred_phs, holo_phs_name)
        save_img(input_img, str.replace(holo_phs_name, 'holo-phase-pred',  'input-img'))
        save_img(input_depth, str.replace(holo_phs_name, 'holo-phase-pred',  'input-depth'))

        # pred_amp_stack = pred_amp_stack.detach().cpu().numpy()
        # pred_amp_stack = pred_amp_stack.transpose((0, 2, 3, 1))
        
        focal_list = []
        for p_id, p_name in [[0, 'far'], [15, 'mid'], [-1, 'near']]:
            pred_image = pred_amp_stack[p_id].unsqueeze(0)
            pred_image = scale(pred_image, rgb_tensor)
            pred_image = torch.clamp(pred_image, 0.0, 1.0)
            pred_image = pred_image.squeeze().detach().cpu().numpy()
            pred_image = pred_image.transpose((1, 2, 0))
            save_img(pred_image,  os.path.join(save_dir, index + f'-fstack-{p_name}-pred.png'))
            focal_list.append(pred_image)
        pred_stack = np.stack(focal_list, axis=0)
        save_fstack_video(pred_stack, os.path.join(save_dir, index + '-focal_stacks.mp4'))



def save_target(test_loader, tr_agent, save_dir, postifx=''):
    psnr_list = []
    psnr_scale_list = [] 
    ssim_list = [] 
    ssim_scale_list = []
    count = 0
    for i, data in enumerate(test_loader):
        batch_size = data[0].size(0)
        start_id = i * batch_size

        for j in range(data[0].size(0)):
            target_amp = data[0]
            target_amp = crop_image(target_amp, [800,1600], True, False)
            target_amp = target_amp[j,...].cpu().squeeze().numpy()
            target_amp = target_amp.transpose((1,2,0))
            index = data[-1][j].split('/')[-1]
            target_name = os.path.join(save_dir, index  + '-target-amp-color.png')            
            save_img(target_amp, target_name) 

        

def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint
    tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    

    mode = 'validate'
    test_loader = get_dataloader(mode, config)
    if config.model_name != '':
        save_dir = "train_log/results/Arbitrary-{}-{}-ckpt-{}".format(mode, config.model_name,config.ckpt)
    else:
        save_dir = "train_log/results/Arbitrary-{}-ckpt-{}".format(mode, config.ckpt) 
    ensure_dir(save_dir)
    # gt_dir = "train_log/results/gt"
    data_folder = '/mnt/data/home/yujie/yujie_data/defocus_data_yujie_new/DEH_TEST_DATA_arbitrary_examples'
    data_files = glob.glob(os.path.join(data_folder, f'scene*/aperture_02/clean_pass_rgb.png'))
    for rgb_path in data_files:
        depth_path = str.replace(rgb_path, 'rgb.png', 'depth.png')
        test(rgb_path, depth_path, tr_agent, save_dir)



if __name__ == '__main__':
    main()


'''
avg_psnr:  21.952550804886744
avg_psnr scale:  29.37540237632974
avg_ssim :  0.8943482221605639
avg_ssim scale:  0.9099432196644597
avg first stage time : 0.012558982372283936
avg_encode time :  23.917447233200072
avg decode time :  26.411681900024416
avg actual bpp : 1.6165295398009945
avg theoretical_bpp bpp :  1.608349347114563
'''
