import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_yy import *
    

class ResnetEncoder(nn.Module):
    def __init__(self, x, num_scale, block_per_scale=1, depth_per_block=2, kernel_size=3, base_dim=16, fc_dim=512, latent_dim=64, second_depth=3, second_dim=1024, cross_entropy_loss=False):
        super(ResnetEncoder, self).__init__()
        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.base_dim = base_dim
        self.fc_dim = fc_dim
        self.x = x
        self.batch_size = x.shape[0]
        self.latent_dim = latent_dim
        self.second_dim = second_dim 
        self.second_depth = second_depth
        self.cross_entropy_loss = cross_entropy_loss

        dim = base_dim
        self.conv0 = nn.Conv2d(x.shape[1], dim, self.kernel_size, 1, cal_padding(x, self.kernel_size, 1))
        self.layers = nn.ModuleList()
        for i in range(self.num_scale):
            self.layers.append(ScaleBlock(x, dim, self.is_training, 'scale'+str(i), self.block_per_scale, self.depth_per_block, self.kernel_size))
            
            if i != self.num_scale - 1:
                dim *= 2
                self.layers.append(DownSample(x, dim, self.kernel_size, 'downsample'+str(i)))

        x = torch.mean(x, [2, 3])
        self.fc = ScaleFcBlock(x, self.fc_dim, 'fc', 1, self.depth_per_block)

        self.mu_z_layer = nn.Linear(x.shape[-1], self.latent_dim)
        self.logsd_z_layer = nn.Linear(x.shape[-1], self.latent_dim)
        

    def forward(self, inputs):
        y = self.conv0(inputs)
        j = 0
        for i in range(self.num_scale):
            y = self.layers[j](y)
            j += 1

            if i +- self.num_scale - 1:
                y = self.layers[j](y)
                j += 1

        y = torch.mean(y, [2, 3])
        y = self.fc(y)

        mu_z = self.mu_z_layer(y)
        logsd_z = self.logsd_z_layer(y)
        sd_z = torch.exp(logsd_z)
        z = mu_z + torch.randn([self.batch_size, self.latent_dim]) * sd_z

        return mu_z, logsd_z, sd_z, z