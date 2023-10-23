from utils_yujie import TrainClock, AverageMeter
import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
import numpy as np
from networks import get_network, set_requires_grad
from tensorboardX import SummaryWriter
from tqdm import tqdm
from losses import *
from model_utils import define_D
from visualization import vis_holo, vis_holo_torch,recon_phaseHolo,recon_sgdHolo_specific
from utils_yujie import interpolate
import torch.nn.functional as F
from losses import XY_loss, lightNess_loss,VGGLoss, GANLoss, computeGrad
import pytorch_ssim
from loss.loss_provider import LossProvider
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import propagation_utils as utils 
from propagator import holo_propagator
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def get_agent(config):
    return HologramAgent(config)


class BaseAgent(object):
    """Base trainer that provides commom training behavior.
        All trainer should be subclass of this class. 
    """
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.device = config.gpu_ids
        self.batch_size = config.batch_size
        self.config = config

        # build network
        self.net, self.netD = self.build_net(config)

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, config):
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().cuda()


    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, gamma=0.98)

        self.optimizerD = optim.Adam(self.netD.parameters(), config.lr_D)
        self.schedulerD = optim.lr_scheduler.StepLR(self.optimizerD, config.lr_step_size, gamma=0.98)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Checkpoint saved at {}".format(save_path))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if isinstance(self.net, nn.DataParallel):
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'modelD_state_dict': self.netD.cpu().state_dict(),
                'optimizerD_state_dict': self.optimizerD.state_dict(),
                'schedulerD_state_dict': self.schedulerD.state_dict()
            }, save_path)
        else:
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'modelD_state_dict': self.netD.cpu().state_dict(),
                'optimizerD_state_dict': self.optimizerD.state_dict(),
                'schedulerD_state_dict': self.schedulerD.state_dict()
            }, save_path)
        self.net.to("cuda:0")
        gpu_ids = self.config.gpu_ids.split(',')
        #self.netD.to("cuda:1")

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Checkpoint loaded from {}".format(load_path))
        if self.config.is_train:
            strict = True 
        else:
            strict = False
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.netD.load_state_dict(checkpoint['modelD_state_dict'], strict=strict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    @abstractmethod
    def forward(self, data):
        pass

    def update_network(self, loss_dict):
        """update network by back propagation"""
        lossG_list = [loss_dict[item] for item in loss_dict.keys() if 'D' not in item and 'eval' not in item]
        lossG = sum(lossG_list)
        self.optimizer.zero_grad()
        lossG.backward()
        self.optimizer.step() 

        lossD_list = [loss_dict[item] for item in loss_dict.keys() if 'D' in item]
        if len(lossD_list) > 0:
            lossD = sum(lossD_list) * 0.5 # (0.5 * (Dreal + Dfake))
            self.optimizerD.zero_grad()
            lossD.backward()
            self.optimizerD.step()




    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step(self.clock.epoch)

        self.train_tb.add_scalar('learning_rate', self.optimizerD.param_groups[-1]['lr'], self.clock.epoch)
        self.schedulerD.step(self.clock.epoch)


    def record_losses(self, loss_dict, mode='train'):
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        # record loss to tensorboard
        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def train_func(self, data):
        """one step of training"""
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)
        self.record_losses(losses, 'train')

        return outputs, losses

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()

        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.record_losses(losses, 'validation')
        return outputs, losses

    def visualize_batch(self, data, mode, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError


class HologramAgent(BaseAgent):
    def build_net(self, config):
        net = get_network(config)
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        gpu_ids = config.gpu_ids.split(',')
        gpu_ids = [int(item) for item in gpu_ids]
        netD = define_D(1, 64, 3,'instance',True, 2, True, gpu_ids)
        self.propagator = holo_propagator(wavelength = self.config.wavelength_list, 
                                    feature_size = self.config.feature_size)
        return net, netD

    def set_loss_function(self):
        self.criterionMSE = nn.MSELoss(size_average=True).cuda()
        self.criterionGrad = Grad_loss().cuda()
        # self.criterionVGG = lightNess_loss().cuda()
        # self.criterionVGG_recon = VGGLoss(device=1)
        # provider = LossProvider()
        # self.criterionWFFT = provider.get_loss_function('watson-fft', deterministic=False, colorspace='grey',
        #                                                      pretrained=True, reduction='sum')
        # self.criterionWFFT = self.criterionWFFT.cuda()
        self.criterionMS_SSIM = ms_ssim
    
    def scale(self, x, label, weight=None):  ##### x.size() should be NxCxHxW
        alpha = torch.cuda.FloatTensor(x.size())

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
    
        if weight is not None:
            x = torch.mul(torch.autograd.Variable(weight,requires_grad=False), x)
            label = torch.mul(torch.autograd.Variable(weight, requires_grad=False), label)

            tensor1 = torch.mul(weight, weight)
            self.weight_normal = tensor1.sum()
            if self.weight_normal != 0:
                self.weight_normal = 1.0 / self.weight_normal
    
        return x

    def recon_focal_stack(self, input_depth, out_amp, out_phs, target_amp, target_phs, recon_dist, beta=0.35):        
        pred_amp_stack, pred_phase_stack = self.propagator(out_phs, recon_dist * 1e-3, out_amp)
        target_amp_stack, target_phase_stack = self.propagator(target_phs, recon_dist * 1e-3, target_amp)
        pred_amp_stack = pred_amp_stack.squeeze()
        target_amp_stack = target_amp_stack.squeeze()

        # create attention maps
        attention_maps = []
        input_depth_transform = self.config.depth_scale * input_depth + self.config.depth_base 
        for d in range(recon_dist.shape[0]): # traverse for all recon distances
            cur_dist = recon_dist[0,d].item() * 1e-3
            #print('cur_dist is ', cur_dist)
            # calculate the attention map at current reconstruction distance 
            cur_att_map = torch.exp(beta * (self.config.depth_scale - torch.abs(cur_dist * 1e3 - input_depth_transform)))
            cur_att_map = cur_att_map / cur_att_map.max()
            attention_maps.append(cur_att_map)
        
        att_map_stack = torch.cat(attention_maps, dim=0)
        #print(pred_amp_stack.shape, target_amp_stack.shape, att_map_stack.shape)
        
        return pred_amp_stack, target_amp_stack, att_map_stack


 
    def forward(self, data):
        input_img, mask, input_depth, target_amp, target_phase, recon_dists, paths = data 
        input_img = input_img.cuda()
        mask = mask.cuda()
        input_depth = input_depth.cuda()
        target_amp = target_amp.cuda()
        target_phase = target_phase.cuda()

        net_input = torch.cat([input_img, input_depth], dim=1)
        net_input = net_input - 0.5 # renormalize the input data

        if not self.config.is_train:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = self.net(net_input)
            end.record()
            torch.cuda.synchronize()
            cur_elapsed = start.elapsed_time(end) / 1000
        else:     
            output = self.net(net_input)

        out_amp = output[:,:3,...] # amplitude
        out_phase = output[:,3:,...] # predicted phase

        # Making sure the values in output_Phase and output_amplitude are in range [-1, 1]
        # print('the range of output amplitude is ', out_amp.max(), out_amp.min())
        # print('the range of output phase is ' ,  out_phase.max(), out_phase.min())

        out_phase = out_phase * np.pi  # from [-1, 1] to [-pi, pi]
        out_amp = (out_amp + 1.0) / 2.0 * np.sqrt(2) # from [-1, 1] to [0, 1]
      
        #print('recon_dists is ', recon_dists)
        scale = self.net.module.s.unsqueeze(0)
        scale = scale.unsqueeze(2).unsqueeze(3)

        cos_diff = torch.cos(out_phase - target_phase)
        cos_diff = torch.where(cos_diff != 0, cos_diff, cos_diff.new(cos_diff.shape).fill_(1e-6))
        delta_phs = torch.atan2(torch.sin(out_phase - target_phase), cos_diff)
        correct_diff_phs = delta_phs - torch.mean(delta_phs, dim=(2,3), keepdim=True) # the phase term (correct phase difference) in Formula (3) of the paper
        correct_cpx = target_amp * torch.exp(1j * correct_diff_phs)
        
        # first calculate the norm for each element and then calculate mean on all element
        data_term = torch.mean(torch.abs(out_amp - correct_cpx), dim=(0,1,2,3))   # formula (3) 

        # out_field_real, out_field_imag = utils.polar_to_rect(out_amp, out_amp.new(out_amp.shape).fill_(0.0))
        # out_field = torch.stack([out_field_real, out_field_imag], dim=-1)

        # calculating reconstruction loss at multiple reconstruction distances               
        all_pcp_loss = torch.tensor([0], dtype=torch.float, requires_grad=True, device='cuda:0')
        # select reconstruction distances
        # reconstruct and calculate the losses for all of the reconstructed images
        for m in range(input_img.size(0)): # enumeate on the batch dimension
            cur_pred_stack, cur_tgt_stack, cur_att_stack = self.recon_focal_stack(input_depth[m,...].unsqueeze(0), 
                                out_amp[m,...].unsqueeze(0), out_phase[m,...].unsqueeze(0), 
                                target_amp[m,...].unsqueeze(0), target_phase[m,...].unsqueeze(0), recon_dists[m].unsqueeze(0))

            cur_pred_intensity = torch.pow(cur_pred_stack, 2)
            cur_tgt_intensity = torch.pow(cur_tgt_stack, 2)
        
            cur_pred_intensity = cur_pred_intensity * scale # scale for minimizing the difference introduced by scale difference.
            #cur_pred_intensity = cur_pred_intensity / (cur_pred_intensity.sum() + 1e-5) * cur_tgt_intensity.sum()
            cur_intensity_loss = torch.abs(cur_pred_intensity - cur_tgt_intensity) 
            cur_grad_loss = torch.abs(computeGrad(cur_pred_intensity) - computeGrad(cur_tgt_intensity))
            cur_recon_loss = torch.mul(cur_intensity_loss + cur_grad_loss, cur_att_stack)
            cur_pcp_loss = torch.mean(torch.abs(cur_recon_loss), dim=(0,1,2,3))
            all_pcp_loss = all_pcp_loss + cur_pcp_loss
        
        all_pcp_loss = all_pcp_loss / input_img.size(0)
        
        #print('data term is ', data_term, 'pcp loss is ', all_pcp_loss)
        loss_dict = {'data_term':  data_term, 'pcp_loss': 20 * all_pcp_loss}

        # return the focal stack reconstructed for the last example in current batch
        output = [out_amp, out_phase, cur_pred_stack, cur_tgt_stack]  

        if not self.config.is_train:
            output.append(cur_elapsed)

        return output, loss_dict

    def infer(self, data):
        self.net.eval()
        data = data.cuda()
        with torch.no_grad():
           output = self.net(data)
        return output

    def visualize_batch(self, data, mode, outputs=None):
        tb = self.train_tb if mode == 'train' else self.val_tb
        input_img = data[0][-1,...] # from bgr to rgb
        input_depth = data[2][-1,0, ...]
        target_amp = data[3][-1, ...] # from bgr to rgb
        target_phs = data[4][-1, ...]
        
        out_amp, out_phs, cur_pred_stack, cur_tgt_stack =  outputs 
        out_amp = out_amp[-1,...].detach().cpu()
        out_phs = out_phs[-1,...].detach().cpu() # from bgr to rgb

        select_indices = random.sample(range(20), 3)
        # select_stacks_pred = cur_pred_stack[select_indices, ...].cpu()
        # select_stacks_tgt = cur_tgt_stack[select_indices, ...].cpu()

        v_line = input_img.new(input_img.size(0), input_img.size(1), 2).fill_(0.0)

        for k in range(3):
            pred_recon_amp = cur_pred_stack[select_indices[k], ...].detach().cpu()
            tgt_recon_amp = cur_tgt_stack[select_indices[k], ...].detach().cpu()        
            full_img = torch.cat([pred_recon_amp, v_line, tgt_recon_amp], dim=2)  # concate along the width dimension
            full_img = torch.clamp(full_img, 0.0, 1.0)
            full_img = self.bgr2rgb(full_img)
            tb.add_image(f'recon_img_{k}', full_img, self.clock.step, dataformats='CHW')

        phase_image = torch.cat([out_phs, v_line, target_phs], dim=2)
        phase_image = (phase_image + np.pi) / (2.0 * np.pi)
        phase_image = torch.clamp(self.bgr2rgb(phase_image), 0.0, 1.0)
        amp_image = torch.cat([out_amp, v_line, target_amp], dim=2)
        amp_image = torch.clamp(self.bgr2rgb(amp_image), 0.0, 1.0)
        input_img = torch.clamp(self.bgr2rgb(input_img), 0.0, 1.0)

        tb.add_image(f'pred_vs_target_amp', amp_image, self.clock.step, dataformats='CHW')
        tb.add_image(f'pred_vs_target_phase', phase_image, self.clock.step, dataformats='CHW')
        tb.add_image(f'input color', input_img, self.clock.step, dataformats='CHW')
        tb.add_image(f'input depth', input_depth, self.clock.step, dataformats='HW')

    def bgr2rgb(self, input):
        temp_input = input.clone()
        input[0,...] = temp_input[2,...]
        input[2,...] = temp_input[0,...]
        return input

        
    