from tqdm import tqdm
import argparse
from dataset import get_dataloader
from common import get_config
from agent import get_agent
from utils_yujie import ensure_dir
import cv2
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

def vis_inputdata(data, save_dir, start_id):
    for j in range(data[0].size(0)):
        sample_name = os.path.join(save_dir, 'test_sample_%d.png'%(start_id+j))
        input_views = data[4:8]
        input_views = [item[j,...] for item in input_views]
        input_view_cat = torch.stack(input_views, dim=0)

        target_view = data[-1][j,...].unsqueeze(0)

        input_view_arr = input_view_cat.numpy() 
        target_view = target_view.numpy() 

        plt.figure()
        plt.scatter(input_view_arr[:, 0], input_view_arr[:, 1], c = 'red', marker='^')
        plt.scatter(target_view[:,0], target_view[:,1], c='blue')
        
        ax=plt.gca()
        x_major_locator=MultipleLocator(0.056)
        y_major_locator=MultipleLocator(0.056)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(-0.28, 0.28)
        plt.ylim(-0.28 ,0.28)
        plt.xticks(rotation = 90)
        plt.tick_params(labelsize=7)
        plt.grid() 
        plt.savefig(sample_name)
        plt.close()


def save_img(img_array, file_name):
    img = (img_array * 255).astype(np.uint8)
    sio.imsave(file_name, img_array)




def test(test_loader, tr_agent, save_dir, postifx=''):
    psnr_list = []
    psnr_scale_list = [] 
    ssim_list = [] 
    ssim_scale_list = []
    count = 0
    writer = open(os.path.join(save_dir, 'errors.txt'), 'a')
    loss_writer = open(os.path.join(save_dir, 'loss_analysis.txt'), 'a')
    writer.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
    loss_writer.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
    loss_all = []
    errors_all = []
    first_time_list = []
    encode_time_list = []
    decode_time_list = []

    for i, data in enumerate(test_loader):
        batch_size = data[0].size(0)
        start_id = i * batch_size
        output, losses = tr_agent.val_func(data)
        first_time_list.append(output[-3])
        encode_time_list.append(output[-2])
        decode_time_list.append(output[-1])

        print('current first stage time is ', output[-3])
        print('current encode time is ',  output[-2])
        print('current decode time is ',  output[-1])
        for j in range(output[0].size(0)):
            target_amp = data[0]
            target_amp = crop_image(target_amp, [800,1600], True, False)
            target_amp = target_amp[j,...].cpu().squeeze().numpy()
            pred_phase = output[1][j,...].detach().cpu().squeeze().numpy()
            recon_img = output[2]
            recon_img = crop_image(recon_img, [800,1600], True, False)
            recon_img = recon_img[j,...].detach().cpu().squeeze().numpy()
            recon_img_scale = output[3]
            recon_img_scale = crop_image(recon_img_scale, [800, 1600], True, False)
            recon_img_scale = recon_img_scale[j,...].detach().cpu().squeeze().numpy()

            if len(recon_img.shape) == 3 and recon_img.shape[0] == 3:
                recon_img = recon_img.transpose((1,2,0))
                recon_img_scale = recon_img_scale.transpose((1,2,0))
                pred_phase = pred_phase.transpose((1,2,0))
                for c in range(3):
                    recon_img[:,:,c] = recon_img[:,:,c]
                    recon_img_scale[:,:,c] = recon_img_scale[:,:,c] 
                    pred_phase[:,:,c] = pred_phase[:,:,c] / np.max(pred_phase[:,:,c])
                target_amp = target_amp.transpose((1,2,0))
                multichannel = True  # for computing ssim value 
            else:
                recon_img = recon_img 
                recon_img_scale = recon_img_scale 
                pred_phase = pred_phase / np.max(pred_phase)
                multichannel = False  # for computing ssim value 
            
            cur_psnr_ori = psnr(target_amp, recon_img)
            cur_psnr_scale = psnr(target_amp, recon_img_scale)
            cur_ssim_ori = ssim(target_amp, recon_img, multichannel=multichannel)
            cur_ssim_scale = ssim(target_amp, recon_img_scale, multichannel=multichannel)
            psnr_list.append(cur_psnr_ori)
            psnr_scale_list.append(cur_psnr_scale)
            ssim_list.append(cur_ssim_ori)
            ssim_scale_list.append(cur_ssim_scale)

            count += 1

            #if count > 5:
            #    break

            index = data[-1][j].split('/')[-1]
            recon_img = recon_img / np.max(recon_img)
            recon_img_scale = recon_img_scale / np.max(recon_img_scale)
            recon_name = os.path.join(save_dir, index + '-recon-img.png')
            recon_name_scale = os.path.join(save_dir, index  + '-recon-img-scale.png')
            phase_name = os.path.join(save_dir, index  + '-pred-phase.png')
            target_name = os.path.join(save_dir, index  + '-target-amp.png')

            loss_all.append({'id': index, 'mse': losses['mse'].item(), 'vgg':losses['vgg'].item(), 'wfft':losses['watson-fft'].item(),
                            'total': losses['mse'].item() + losses['vgg'].item(),
                            'psnr': cur_psnr_scale, 'ssim': cur_ssim_scale})
            errors_all.append({'id': index, 'psnr':cur_psnr_ori, 'psnr_scale':cur_psnr_scale, 
                            'ssim':cur_ssim_ori, 'ssim_scale':cur_ssim_scale})


            print(recon_name)
            save_img(recon_img,recon_name)
            save_img(recon_img_scale, recon_name_scale)
            save_img(pred_phase, phase_name)
            save_img(target_amp, target_name) 
        #if count > 6:
        #    break

    errors_all = sorted(errors_all, key=itemgetter('psnr_scale'), reverse=True)
    for item in errors_all:
        writer.write('Index: %s \n' % (item['id']))
        writer.write('PSNR %.4f  PSNR (scale) %.4f   SSIM %.4f  SSIM (scale) %.4f \n'
                                 %(item['psnr'], item['psnr_scale'], item['ssim'], item['ssim_scale']))
    avg_psnr = sum(psnr_list) / count 
    avg_psnr_scale = sum(psnr_scale_list) / count
    avg_ssim = sum(ssim_list) / count 
    avg_ssim_scale = sum(ssim_scale_list) / count
    avg_first_time = sum(first_time_list) / count 
    avg_encode_time = sum(encode_time_list) / count 
    avg_decode_time = sum(decode_time_list) / count 
    
    print('avg_psnr: ',  avg_psnr)
    print('avg_psnr scale: ', avg_psnr_scale)
    print('avg_ssim : ', avg_ssim)
    print('avg_ssim scale: ',avg_ssim_scale)
    print('avg first stage time :', avg_first_time)
    print('avg_encode time : ', avg_encode_time)
    print('avg decode time : ', avg_decode_time)
        


    writer.write('avg PSNR: %.4f \n'%(avg_psnr))
    writer.write('avg PSNR (scale): %.4f  \n'%(avg_psnr_scale))
    writer.write('avg SSIM: %.4f \n'%(avg_ssim))
    writer.write('avg SSIM (scale): %.4f \n'%(avg_ssim_scale))
    writer.write('avg first stage time %.4f'%(avg_first_time))
    writer.write('avg encode time %.4f'%(avg_encode_time))
    writer.write('avg decode time %.4f'%(avg_decode_time))
    writer.close()


    loss_all = sorted(loss_all, key=itemgetter('total'), reverse=True)
    for item in loss_all:
        loss_writer.write('id: %s total (mse + vgg):  %.4f  vgg: %.4f  mse: %.4f wfft: %.4f,  psnr: %.4f  ssim: %.4f\n'%(item['id'], 
                        item['total'], item['vgg'], item['mse'], item['wfft'] * 1e9, item['psnr'], item['ssim']))
    loss_writer.close()



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

'''
red channel 
epoch 6:
avg_psnr: 23.9377
avg_psnr_scale: 23.9373
avg_ssim: 0.7658
avg_ssim_scale: 0.7658

epoch 18
avg_psnr:  20.368407431963796
avg_psnr scale:  23.937350272577902
avg_ssim :  0.7733523555510505
avg_ssim scale:  0.7840976792410129
'''
        

def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint
    tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    

    mode = 'test'
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
