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
    img = (img_array * 255).astype(np.uint8)
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


def test(test_loader, tr_agent, save_dir, postifx=''):
    psnr_list = []
    ssim_list = [] 
    time_list = []
    count = 0
    writer = open(os.path.join(save_dir, 'errors.txt'), 'a')
    loss_writer = open(os.path.join(save_dir, 'loss_analysis.txt'), 'a')
    
    writer.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
    errors_all = []

    for i, data in enumerate(test_loader):
        batch_size = data[0].size(0)
        start_id = i * batch_size
        outputs, losses = tr_agent.val_func(data)
        out_amp, out_phase, pred_stack, tgt_stack, cur_elapsed = outputs
        
        print('shape of pred_stack and tgt_stack are ', pred_stack.shape, tgt_stack.shape)

        for j in range(batch_size):
            time_list.append(cur_elapsed / batch_size)
            target_amp = data[3][j, ...].squeeze().detach().cpu().numpy()
            target_phs = data[4][j, ...].squeeze().detach().cpu().numpy()

            pred_amp = out_amp[j, ...].squeeze().detach().cpu().numpy()
            pred_phs = out_phase[j, ...].squeeze().detach().cpu().numpy()
            

            if len(target_amp.shape) == 3 and target_amp.shape[0] == 3:
                target_amp = target_amp.transpose((1,2,0))
                target_phs = target_phs.transpose((1,2,0))
                pred_amp = pred_amp.transpose((1,2,0))
                pred_phs = pred_phs.transpose((1,2,0))
                multichannel = True  # for computing ssim value 
            else:
                multichannel = False  # for computing ssim value 

            target_amp = np.clip(target_amp, 0.0, 1.0)
            pred_amp = np.clip(pred_amp, 0.0, 1.0)
            
            cur_psnr_amp = psnr(target_amp, pred_amp)
            cur_ssim_amp = ssim(target_amp, pred_amp, multichannel=multichannel)
         
            psnr_list.append(cur_psnr_amp)
            ssim_list.append(cur_ssim_amp)

            count += 1

            index = data[-1][j].split('/')[-1][:-4]

            # calculate errors for the predicted focal stacks
            errors_fstack = cal_fstack_error(pred_stack, tgt_stack)
            errors_all.append({'id': index, 'psnr_amp':cur_psnr_amp, 'ssim_amp': cur_ssim_amp, 
                               'psnr_fstack': errors_fstack[4], 'ssim_fstack': errors_fstack[5],
                               'psnr_fstack_scale': errors_fstack[6], 'ssim_fstack_scale': errors_fstack[7],
                               'psnr_list': errors_fstack[0], 'ssim_list': errors_fstack[1],
                               'psnr_list_scale': errors_fstack[2], 'ssim_list_scale': errors_fstack[3]})

            # save the produced images or videos
            pred_phs = (pred_phs + np.pi ) / (2 * np.pi) 
            target_phs = (target_phs + np.pi) / (2 * np.pi)
            
            holo_amp_name = os.path.join(save_dir, index + '-holo-amp-pred.png')
            holo_phs_name = os.path.join(save_dir, index  + '-holo-phase-pred.png')

            save_img(pred_amp, holo_amp_name)
            save_img(pred_phs, holo_phs_name)
            save_img(target_amp, str.replace(holo_amp_name, 'pred', 'tgt'))
            #print('target_phs.shape is', target_phs.shape)
            save_img(target_phs, str.replace(holo_phs_name, 'pred', 'tgt')) 

            cat_stack = errors_fstack[-1]
            save_fstack_video(cat_stack, os.path.join(save_dir, index + '-focal_stacks.mp4'))

    errors_all = sorted(errors_all, key=itemgetter('psnr_fstack_scale'), reverse=True)
    for item in errors_all:
        writer.write('Index: %s \n' % (item['id']))
        writer.write('\t PSNR_amp: %.4f  SSIM_amp: %.4f  PSNR_fstack: %.4f  SSIM_fstack: %.4f  PSNR_fs_scale: %.4f SSIM_fs_scale: %.4f \n'
                                 %(item['psnr_amp'], item['ssim_amp'], 
                                   item['psnr_fstack'], item['ssim_fstack'], 
                                   item['psnr_fstack_scale'], item['ssim_fstack_scale']))
        for m in range(len(item['psnr_list_scale'])):
            writer.write('\t\t Plane-%d: PSNR: %.4f  SSIM: %.4f PSNR_scale: %.4f  SSIM_scale: %.4f \n' %(m,
                                item['psnr_list'][m], item['ssim_list'][m], item['psnr_list_scale'][m], item['ssim_list_scale'][m]))
    
    
    writer.write('AVG VALUES\n')
    
    for key in errors_all[0].keys():
        if 'list' not in key and key != 'id':
            cur_error_list = [item[key] for item in errors_all]
            avg_value = sum(cur_error_list) / len(cur_error_list)
            print(f'avg {key} : {avg_value:.4f}')
            writer.write(f'avg {key} : {avg_value:.4f}\n')
    
    avg_elapsed = sum(time_list) / count
    writer.write(f'AVG ELAPSED: {avg_elapsed:.4f}\n')
    print(f'AVG ELAPSED: {avg_elapsed:.4f} seconds')

    writer.write(f'Count: {count:d}\n')
    print(f'Count: {count:d}\n')
    writer.close()


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
        save_dir = "train_log/results/{}-{}-ckpt-{}".format(mode, config.model_name,config.ckpt)
    else:
        save_dir = "train_log/results/{}-ckpt-{}".format(mode, config.ckpt) 
    ensure_dir(save_dir)
    # gt_dir = "train_log/results/gt"
    test(test_loader, tr_agent, save_dir)



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
