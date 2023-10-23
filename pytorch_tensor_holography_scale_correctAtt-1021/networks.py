import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Functions
##############################################################################
def get_network(config):
    net = Model(config.n_layers, config.input_dim, config.output_dim, config.inter_dim, 
                config.kernel_size, config.norm_type)
    return net

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] =  grad 
    return hook

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

################################################################################
class Residual_block(nn.Module):
    def __init__(self, in_channels, inter_channels, kernel_size=3, norm='batch'):
        super(Residual_block, self).__init__()
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        else: 
            norm_layer = nn.InstanceNorm2d
        conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size, 1, padding=int(kernel_size//2))
        self.layer1 = nn.Sequential(conv1, norm_layer(inter_channels), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, kernel_size, 1, padding=int(kernel_size//2)), 
                                    norm_layer(inter_channels), nn.ReLU())

    def forward(self, input_data):
        out1 = self.layer1(input_data)
        out2 = self.layer2(out1)
        out = input_data + out2 
        return out 

class Model(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, inter_dim, kernel_size=3, norm='instance'):
        super(Model, self).__init__()
        self.n_layers = n_layers 
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.inter_dim = inter_dim 
        self.s_r = nn.Parameter
        s = torch.tensor([0.95, 0.95, 0.95], requires_grad=True)
        self.s = torch.nn.Parameter(s)


        for i in range(self.n_layers):
            cur_outDim = self.inter_dim 
            if i == 0 :
                in_dim = self.input_dim 
                cur_layer = nn.Sequential(nn.Conv2d(in_dim, cur_outDim, kernel_size, stride=1, padding=1), 
                                nn.BatchNorm2d(cur_outDim), nn.ReLU())
            else: 
                if i < (self.n_layers - 1):
                    in_dim = self.inter_dim 
                    cur_layer = Residual_block(in_dim, cur_outDim, kernel_size, norm)
                else: 
                    cur_layer = nn.Sequential(nn.Conv2d(int(self.input_dim + self.inter_dim), self.output_dim, kernel_size=kernel_size, 
                                             stride=1, padding=1), nn.Tanh())
            setattr(self, f'layer_{i:d}', cur_layer)
    
    def forward(self, input_data): 
        out = input_data
        for i in range(self.n_layers):
            cur_layer = getattr(self, f'layer_{i:d}')
            if i != (self.n_layers - 1): 
                out = cur_layer(out)
            else:
                skip_out = torch.cat([input_data, out], dim=1)
                out = cur_layer(skip_out)
        return out


if __name__ == '__main__':
    import torch
    model = Model(30, 4, 6, 24, 3)
    print(model)
    model = model.cuda()
    input_data = torch.rand(2, 4, 192, 192)
    target = torch.rand(2, 6, 192, 192)
    input_data = input_data.cuda() 
    target = target.cuda() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(100):
        optimizer.zero_grad()
        result = model(input_data)
        loss = torch.nn.functional.mse_loss(result, target)
        loss.backward()
        optimizer.step()
        print(f'iteration {i:d}, loss is  {loss.item():.4f}.')
        