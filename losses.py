import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
import numpy as np
from model_utils import Vgg19, Vgg19_conv4
from torch.autograd import Variable
import random 
import torch.nn.functional as F


#################### functions########################
def computeGrad(input):
    input_padY = torch.nn.functional.pad(input, (0,0,0,1), 'constant')
    input_padX = torch.nn.functional.pad(input, (0,1,0,0), 'constant') 

    input_diffx = torch.abs(input_padX[:,:,:,1:] - input_padX[:,:,:,:-1])
    input_diffy = torch.abs(input_padY[:,:,1:,:] - input_padY[:,:,:-1,:])
    grad_map = torch.norm(torch.stack([input_diffx, input_diffy], dim=1)+1e-6, p=2, dim=1)
    return grad_map
##################### classes ########################

class lightNess_loss(nn.Module):
    def __init__(self):
        super(lightNess_loss, self).__init__()        
        self.vgg = Vgg19_conv4().cuda()
        self.criterion = nn.MSELoss()     

    def forward(self, x, y):       
        if x.size(1) == 1:
            x = x.repeat(1,3,1,1)
        if  y.size(1) == 1:
            y = y.repeat(1,3,1,1)       
        x_vgg_conv4, y_vgg_conv4 = self.vgg(x), self.vgg(y)
        loss = self.criterion(x_vgg_conv4, y_vgg_conv4.detach())        
        return loss


class siMSELoss(nn.Module):
    def __init__(self, scale_invariant=True):
        super(siMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.scale_invariant = scale_invariant
        self.weight_normal = 1.0

    def compute_loss(self, x, label, weight=None):  ##### x.size() should be NxCxHxW
        if self.scale_invariant:
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
        

        loss = self.criterion(x,label)
        return loss

    def __call__(self, pred, target):  #### targets should contain [label, weight_map]
        '''
        label = targets[0]
        weight = targets[1]
        '''
        result_loss = self.compute_loss(pred, target)
        return result_loss


class Grad_loss(nn.Module):
    def __init__(self):
        super(Grad_loss,self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input_padY = torch.nn.functional.pad(input, (0,0,0,1), 'constant')
        input_padX = torch.nn.functional.pad(input, (0,1,0,0), 'constant') 
      
        target_padY = torch.nn.functional.pad(target, (0,0,0,1), 'constant')
        target_padX = torch.nn.functional.pad(target, (0,1,0,0), 'constant') 

        input_padX = torch.abs(input_padX)
        input_padY = torch.abs(input_padY)
        target_padX = torch.abs(target_padX)
        target_padY = torch.abs(target_padY)

        input_diffx = (input_padX[:,:,:,1:]) / (input_padX[:,:,:,:-1] +  1e-5)
        input_diffy = (input_padY[:,:,1:,:]) / (input_padY[:,:,:-1,:] + 1e-5)
        target_diffx = (target_padX[:,:,:,1:]) / (target_padX[:,:,:,:-1] + 1e-5)
        target_diffy = (target_padY[:,:,1:,:]) / (target_padY[:,:,:-1,:] + 1e-5)

        grad_map_input = torch.norm(torch.stack([input_diffx, input_diffy], dim=1), p=2, dim=1)
        grad_map_target = torch.norm(torch.stack([target_diffx, target_diffy], dim=1), p=2, dim=1)
        grad_map_input = grad_map_input / torch.max(torch.max(grad_map_input, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        grad_map_target = grad_map_target / torch.max(torch.max(grad_map_target, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        grad_mask = torch.lt(torch.abs(grad_map_target-1), 0.1)

        loss = torch.mean(torch.mul(torch.abs(grad_map_input - grad_map_target), grad_mask), dim=(0,1,2,3))

        return loss 

    def compute_grad(self, x):
        x_diffx = x[:,:,:,1:] - x[:,:,:,:-1]
        x_diffy = x[:,:,1:,:] - x[:,:,:-1,:]

        return x_diffx, x_diffy

#### VGG Loss, borrowed from Pix2pixHD
#### https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
#####

class VGGLoss(nn.Module):
    def __init__(self, device=0):
        super(VGGLoss, self).__init__()   
        self.device = torch.device("cuda:%d"%(device))
        self.vgg = Vgg19().to(self.device)
        self.criterion = nn.L1Loss()
        #self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]      

    def forward(self, x, y):   
        if x.size(1) == 1 and y.size(1) == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1) 
        if self.device != "cuda:0":
            x = x.to(self.device)
            y = y.to(self.device) 
        #print(x.device)     
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = input.new(input.size()).fill_(self.real_label)
                self.real_label_var.requires_grad=False
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = input.new(input.size()).fill_(self.fake_label)
                self.fake_label_var.requires_grad=False
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class XY_loss(nn.Module):
    def __init__(self, n_samples=10):
        super(XY_loss, self).__init__()
        self.n_samples = n_samples
        self.criterion = Grad_loss()

    def forward(self, data, output_recon):
        target_img_recon = data[7]
        target_slice = data[6]
        
        all_slices  = data[-1]
        input_recons = data[-3].to(output_recon.device)

        batch_size, channel, height, width = input_recons.size()
        if output_recon.size(2) >  height:
            output_recon = output_recon[:,:,height//2:height//2+height, width//2:width//2+width]

        all_slices = torch.cat([all_slices, target_slice], dim=1)
        all_recons_pred = torch.cat([input_recons, output_recon], dim=1)
        all_recons = torch.cat([input_recons, target_img_recon], dim=1)
        

        all_slices = all_slices.long()

        x_slices = all_recons.new(all_recons.size(0), self.n_samples, all_recons.size(2), all_recons.size(3)).fill_(0.0)
        y_slices = all_recons.new(all_recons.size(0), self.n_samples, all_recons.size(2), all_recons.size(3)).fill_(0.0)
        x_slices_pred = all_recons.new(all_recons.size(0), self.n_samples, all_recons.size(2), all_recons.size(3)).fill_(0.0)
        y_slices_pred = all_recons.new(all_recons.size(0), self.n_samples, all_recons.size(2), all_recons.size(3)).fill_(0.0)
        x_range = list(range(10, all_recons.size(3)-10))
        y_range = list(range(10, all_recons.size(2)-10))

        for n in range(batch_size):
            x_select = torch.Tensor(random.sample(x_range, self.n_samples)).view(1,self.n_samples).long().to(output_recon.device)
            y_select = torch.Tensor(random.sample(y_range, self.n_samples)).view(1,self.n_samples).long().to(output_recon.device)
            x_samples = torch.index_select(all_recons[n,...], dim=2, index=x_select[0,:]).permute(0,2,1).permute(1,0,2).permute(0,2,1).contiguous()
            y_samples = torch.index_select(all_recons[n,...], dim=1, index=y_select[0,:]).permute(1,0,2).permute(0,2,1).contiguous()
            x_samples_pred = torch.index_select(all_recons_pred[n,...], dim=2, index=x_select[0,:]).permute(0,2,1).permute(1,0,2).permute(0,2,1).contiguous()
            y_samples_pred = torch.index_select(all_recons_pred[n,...], dim=1, index=y_select[0,:]).permute(1,0,2).permute(0,2,1).contiguous()
            x_slices[n,:,:,all_slices[n,:]] = x_samples 
            y_slices[n,:,:,all_slices[n,:]] = y_samples
            x_slices_pred[n,:,:,all_slices[n,:]] = x_samples_pred 
            y_slices_pred[n,:,:,all_slices[n,:]] = y_samples_pred

        x_slice_gradX, x_slice_gradY = self.criterion.compute_grad(x_slices)
        x_slice_gradX_pred, x_slice_gradY_pred = self.criterion.compute_grad(x_slices_pred)
        y_slice_gradX, y_slice_gradY = self.criterion.compute_grad(y_slices)
        y_slice_gradX_pred, y_slice_gradY_pred = self.criterion.compute_grad(y_slices_pred)

        '''
        compute gradient along channel dimension, which means computing gradients between shuffled arrays of x-direction and y-direction 
        since there is only one column coming from prediction results (all other columns are zeros or data coming from input data), so there only one array has errors 
        so I just divide the result with height or width.
        '''
        xslice_gradC = x_slices[:,1:,...] - x_slices[:,:-1,...]
        xslice_gradC_pred = x_slices_pred[:,1:,...] - x_slices_pred[:,:-1,...]
        yslice_gradC = y_slices[:,1:,...] - y_slices[:,:-1,...]
        yslice_gradC_pred = y_slices_pred[:,1:,...] - y_slices_pred[:,:-1,...]

        x_grad_loss =  (torch.sum(torch.abs(x_slice_gradX_pred - x_slice_gradX),dim=1).mean() +
                        torch.sum(torch.abs(x_slice_gradY_pred - x_slice_gradY),dim=1).mean() )
        y_grad_loss =  (torch.sum(torch.abs(y_slice_gradX_pred - y_slice_gradX),dim=1).mean() +
                        torch.sum(torch.abs(y_slice_gradY_pred - y_slice_gradY),dim=1).mean() )
        x_gradC_loss =  torch.abs(xslice_gradC_pred - xslice_gradC).sum() / width
        y_gradC_loss =  torch.abs(yslice_gradC_pred - yslice_gradC).sum() / height

        total_loss = x_grad_loss + y_grad_loss  + x_gradC_loss + y_gradC_loss
        return total_loss