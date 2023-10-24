'''A number of custom pytorch modules with sane defaults that I find useful for model prototyping.'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils

import numpy as np

import math
import numbers
from utils.pytorch_prototyping import channel as channel

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.net(input)


# From https://gist.github.com/wassname/ecd2dac6fc8f9918149853d17e3abf02
class LayerNormConv2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self,item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


class DownBlock3D(nn.Module):
    '''A 3D convolutional downsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm3d):
        super().__init__()

        self.net = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=4,
                      padding=0,
                      stride=2,
                      bias=False if norm is not None else True),
        ]

        if norm is not None:
            self.net += [norm(out_channels, affine=True)]

        self.net += [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class UpBlock3D(nn.Module):
    '''A 3D convolutional upsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm3d):
        super().__init__()

        self.net = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False if norm is not None else True),
        ]

        if norm is not None:
            self.net += [norm(out_channels, affine=True)]

        self.net += [nn.ReLU(True)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)


class Conv3dSame(torch.nn.Module):
    '''3D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReplicationPad3d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb, ka, kb)),
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

    def forward(self, x):
        return self.net(x)


class Conv2dSame(torch.nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    '''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
    reasonable defaults. (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param post_conv: Whether to have another convolutional layer after the upsampling layer.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param upsampling_mode: Which upsampling mode:
                transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
                bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
        '''
        super().__init__()

        net = list()

        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=True if norm is None else False)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [
                Conv2dSame(in_channels // 4, out_channels, kernel_size=3,
                           bias=True if norm is None else False)]
        else:
            raise ValueError("Unknown upsampling mode!")

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.ReLU(True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        if post_conv:
            net += [Conv2dSame(out_channels,
                               out_channels,
                               kernel_size=3,
                               bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(out_channels, affine=True)]

            net += [nn.ReLU(True)]

            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)


class DownBlock(nn.Module):
    '''A 2D-conv downsampling block following best practices / with reasonable defaults
    (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param prep_conv: Whether to have another convolutional layer before the downsampling layer.
        :param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
                                convs.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        if middle_channels is None:
            middle_channels = in_channels

        net = list()

        if prep_conv:
            net += [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels,
                              middle_channels,
                              kernel_size=3,
                              padding=0,
                              stride=1,
                              bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(middle_channels, affine=True)]

            net += [nn.LeakyReLU(0.2, True)]

            if use_dropout:
                net += [nn.Dropout2d(dropout_prob, False)]

        net += [nn.ReflectionPad2d(1),
                nn.Conv2d(middle_channels,
                          out_channels,
                          kernel_size=4,
                          padding=0,
                          stride=2,
                          bias=True if norm is None else False)]

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Unet3d(nn.Module):
    '''A 3d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 norm=nn.BatchNorm3d,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet3d."

        # Define the in block
        self.in_layer = [Conv3dSame(in_channels, nf0, kernel_size=3, bias=False)]

        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]

        self.in_layer += [nn.LeakyReLU(0.2, True)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block. The feature map has height and width 1 --> no batchnorm.
        self.unet_block = UnetSkipConnectionBlock3d(int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                                    int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                                    norm=None)
        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock3d(int(min(2 ** i * nf0, max_channels)),
                                                        int(min(2 ** (i + 1) * nf0, max_channels)),
                                                        submodule=self.unet_block,
                                                        norm=norm)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv3dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear)]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


class UnetSkipConnectionBlock3d(nn.Module):
    '''Helper class for building a 3D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 norm=nn.BatchNorm3d,
                 submodule=None):
        super().__init__()

        if submodule is None:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     UpBlock3D(inner_nc, outer_nc, norm=norm)]
        else:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     submodule,
                     UpBlock3D(2 * inner_nc, outer_nc, norm=norm)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)


class UnetSkipConnectionBlock(nn.Module):
    '''Helper class for building a 2D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super().__init__()

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)


class Unet(nn.Module):
    '''A 2d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 use_dropout,
                 upsampling_mode='transpose',
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param use_dropout: Whether to use dropout or no.
        :param dropout_prob: Dropout probability if use_dropout=True.
        :param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in block
        self.in_layer = [Conv2dSame(in_channels, nf0, kernel_size=3, bias=True if norm is None else False)]
        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels),
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  use_dropout=use_dropout,
                                                  dropout_prob=dropout_prob,
                                                  norm=None, # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)

        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      use_dropout=use_dropout,
                                                      dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv2dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear or (norm is None))]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]

            if use_dropout:
                self.out_layer += [nn.Dropout2d(dropout_prob)]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer_weight = self.out_layer[0].weight

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


class Identity(nn.Module):
    '''Helper module to allow Downsampling and Upsampling nets to default to identity if they receive an empty list.'''

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class DownsamplingNet(nn.Module):
    '''A subnetwork that downsamples a 2D feature map with strided convolutions.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 use_dropout,
                 out_channels=None,
                 dropout_prob=0.1,
                 last_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of downsampling steps (each step dowsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param last_layer_one: Whether the output of the last layer will have a spatial size of 1. In that case,
                               the last layer will not have batchnorm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()

        if not len(per_layer_out_ch):
            self.downs = Identity()
        else:
            self.downs = list()
            self.downs.append(DownBlock(in_channels, per_layer_out_ch[0], use_dropout=use_dropout,
                                        dropout_prob=dropout_prob, middle_channels=per_layer_out_ch[0], norm=norm))
            for i in range(0, len(per_layer_out_ch) - 1):
                if last_layer_one and (i == len(per_layer_out_ch) - 2):
                    norm = None
                self.downs.append(DownBlock(per_layer_out_ch[i],
                                            per_layer_out_ch[i + 1],
                                            dropout_prob=dropout_prob,
                                            use_dropout=use_dropout,
                                            norm=norm))
        if  out_channels != None:
            self.downs.append(Conv2dSame(per_layer_out_ch[-1], out_channels, kernel_size=3, bias = True))
            self.downs = nn.Sequential(*self.downs)

    def forward(self, input):
        return self.downs(input)


class UpsamplingNet(nn.Module):
    '''A subnetwork that upsamples a 2D feature map with a variety of upsampling options.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 upsampling_mode,
                 use_dropout,
                 dropout_prob=0.1,
                 first_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of upsampling steps (each step upsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param upsampling_mode: Mode of upsampling. For documentation, see class "UpBlock"
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param first_layer_one: Whether the input to the last layer will have a spatial size of 1. In that case,
                               the first layer will not have a norm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()

        if not len(per_layer_out_ch):
            self.ups = Identity()
        else:
            self.ups = list()
            self.ups.append(UpBlock(in_channels,
                                    per_layer_out_ch[0],
                                    use_dropout=use_dropout,
                                    dropout_prob=dropout_prob,
                                    norm=None if first_layer_one else norm,
                                    upsampling_mode=upsampling_mode))
            for i in range(0, len(per_layer_out_ch) - 1):
                self.ups.append(
                    UpBlock(per_layer_out_ch[i],
                            per_layer_out_ch[i + 1],
                            use_dropout=use_dropout,
                            dropout_prob=dropout_prob,
                            norm=norm,
                            upsampling_mode=upsampling_mode))
            self.ups = nn.Sequential(*self.ups)

    def forward(self, input):
        return self.ups(input)


#########################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, 
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = getattr(F, activation)
        #norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        norm_kwargs = dict(affine=True)

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            #self.interlayer_norm = instance.InstanceNorm2D_wrap
            self.interlayer_norm = nn.BatchNorm2d

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res) 
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)


def InstanceNorm2D_wrap(input_channels, momentum=0.1, affine=True,
                        track_running_stats=False, **kwargs):
    """ 
    Wrapper around default Torch instancenorm
    """
    instance_norm_layer = nn.InstanceNorm2d(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)
    return instance_norm_layer

class Generator(nn.Module):
    def __init__(self, filters=[960, 480, 240, 120, 60], in_channels=16, activation='relu',
                 n_residual_blocks=8, channel_norm=True, sample_noise=False,
                 noise_dim=32):

        """ 
        Generator with convolutional architecture proposed in [1].
        Upscales quantized encoder output into feature map of size C x W x H.
        Expects input size (C,16,16)
        ========
        Arguments:
        input_dims: Dimensions of quantized representation, (C,H,W)
        batch_size: Number of instances per minibatch
        C:          Encoder bottleneck depth, controls bits-per-pixel
                    C = 220 used in [1].

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Generator, self).__init__()
        
        kernel_dim = 3
        self.n_residual_blocks = n_residual_blocks
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        #norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        norm_kwargs = dict(affine=True)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_upsampling_layers = len(filters) - 1

        
        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            #self.interlayer_norm = instance.InstanceNorm2D_wrap
            self.interlayer_norm = nn.BatchNorm2d

        self.pre_pad = nn.ReflectionPad2d(1)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(3)

        # (16,16) -> (16,16), with implicit padding
        self.conv_block_init = nn.Sequential(
            self.interlayer_norm(in_channels, **norm_kwargs),
            self.pre_pad,
            nn.Conv2d(in_channels, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
        )

        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(in_channels=filters[0], 
                channel_norm=channel_norm, activation=activation)
            self.add_module(f'resblock_{str(m)}', resblock_m)
        
        for i in range(0, len(filters)-1):
            cur_block = nn.Sequential(
                nn.ConvTranspose2d(filters[i], filters[i+1], kernel_dim, **cnn_kwargs),
                self.interlayer_norm(filters[i+1], **norm_kwargs),
                self.activation(),
            )
            setattr(self, 'upconv_block%d'%(i+1), cur_block)


        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[-1], 1, kernel_size=(7,7), stride=1),
        )


    def forward(self, x):        
        head = self.conv_block_init(x)

        if self.sample_noise is True:
            B, C, H, W = tuple(head.size())
            z = torch.randn((B, self.noise_dim, H, W)).to(head)
            head = torch.cat((head,z), dim=1)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)
        
        x += head
        for k in range(self.n_upsampling_layers):
            cur_layer = getattr(self, 'upconv_block%d'%(k+1))
            x = cur_layer(x)

        out = self.conv_block_out(x)

        return out



class MultiScale_Encoder(nn.Module):
    '''A subnetwork that downsamples a 2D feature map with strided convolutions.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 use_dropout,
                 out_channels=None,
                 dropout_prob=0.1,
                 last_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of downsampling steps (each step dowsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param last_layer_one: Whether the output of the last layer will have a spatial size of 1. In that case,
                               the last layer will not have batchnorm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()
        self.per_layer_out_ch = per_layer_out_ch

        if not len(per_layer_out_ch):
            self.downs = Identity()
        else:
            self.down_conv0 = DownBlock(in_channels, per_layer_out_ch[0], use_dropout=use_dropout,
                                        dropout_prob=dropout_prob, middle_channels=per_layer_out_ch[0], norm=norm)
            for i in range(0, len(per_layer_out_ch) - 1):
                if last_layer_one and (i == len(per_layer_out_ch) - 2):
                    norm = None
                cur_layer = [DownBlock(per_layer_out_ch[i],
                                            per_layer_out_ch[i + 1],
                                            dropout_prob=dropout_prob,
                                            use_dropout=use_dropout,
                                            norm=norm)]
                cur_layer.append(ResidualBlock(in_channels=per_layer_out_ch[i + 1]))
                setattr(self, 'down_conv%d'%(i+1), nn.Sequential(*cur_layer))

            
            self.direct_down1 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=4,
                                  padding=0,
                                  stride=4,
                                  bias=True if norm is None else False))

            self.direct_down2 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=8,
                                  padding=0,
                                  stride=8,
                                  bias=True if norm is None else False),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))

            smooth_layer = nn.Sequential(
                                Conv2dSame(int(3 * per_layer_out_ch[-1]), per_layer_out_ch[-1], kernel_size=3, bias = None),
                                norm(per_layer_out_ch[-1]),
                                nn.LeakyReLU(0.2))

            setattr(self, 'fusion_layer', smooth_layer)
            out_resLayer = ResidualBlock(in_channels= int(per_layer_out_ch[-1]))
            setattr(self, 'out_res', out_resLayer)

        if  out_channels != None:
            out_layer = Conv2dSame(int(per_layer_out_ch[-1]), out_channels, kernel_size=3, bias = True)
            setattr(self, 'out_layer', out_layer)

    def forward(self, input):
        out_down1 = self.direct_down1(input)
        out_down2 = self.direct_down2(input)
        x = input 
        for k in range(len(self.per_layer_out_ch)):
            cur_layer = getattr(self, 'down_conv%d'%(k))
            x = cur_layer(x)
        cat_features = torch.cat([out_down1, out_down2, x], dim=1)
        out = self.fusion_layer(cat_features)
        out = self.out_res(out)
        out = self.out_layer(out)
        return out
