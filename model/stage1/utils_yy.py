import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Iterable

def cal_padding(x, kernel_size, stride=1):
    m = []
    s = []
    if isinstance(kernel_size, Iterable):
        m = kernel_size
    else:
        m[0] = m[1] = kernel_size
    if isinstance(stride, Iterable):
        s = stride
    else:
        s[0] = s[1] = stride
    n = x.shape[-2:]
    return [(((-n[i]) % s[i] - 1) * s[i] + m[i] + 1) // 2 for i in range(2)]


class ResBlock(nn.Module):
    def __init__(self, x, out_dim, is_training, name, depth=2, kernel_size=3):
        super(ResBlock, self).__init__()
        self.depth = depth
        self.name = name
        
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.conv2ds = nn.ModuleList()
        input_channel = x.shape[1]
        for _ in range(depth):
            self.batch_norms.append(nn.BatchNorm2d(input_channel))
            self.relus.append(nn.ReLU())
            self.conv2ds.append(nn.Conv2d(input_channel, input_channel, kernel_size, padding=cal_padding(x, kernel_size)))
        self.s = nn.Conv2d(input_channel, out_dim, kernel_size, padding=(x, kernel_size))

    def forward(self, x):
        y = x
        for i in range(self.depth):
            y = self.batch_norms[i](y)
            y = self.relus[i](y)
            y = self.conv2ds[i](y)
        y = y + self.s(x)
        return y
    
    def string(self):
        return self.name


class ScaleBlock(nn.Module):
    def __init__(self, x, out_dim, is_training, name, block_per_scale=1, depth_per_block=2, kernel_size=3):
        super(ScaleBlock, self).__init__()
        self.block_per_scale = block_per_scale
        self.ResBlocks = nn.ModuleList()
        for i in range(block_per_scale):
            self.ResBlocks.append(ResBlock(x, out_dim, is_training, 'block'+str(i), depth_per_block, kernel_size))

    def forward(self, x):
        y = x
        for i in range(self.block_per_scale):
            y = self.ResBlocks[i](y)
        return y

    def string(self):
        return self.name


class DownSample(nn.Module):
    def __init__(self, x, out_dim, kernel_size, name):
        self.name = name
        assert(len(x.shape) == 4)
        self.layer = nn.Conv2d(x.shape[1], out_dim, kernel_size, 2, cal_padding(x, kernel_size, 2))

    def forward(self, x):
        return self.layer(x)

    def string(self):
        return self.name


class UpSample(nn.Module):
    def __init__(self, x, out_dim, kernel_size, name):
        self.name = name
        assert(len(x.shape) == 4)
        self.layer = nn.ConvTranspose2d(x.shape[1], out_dim, kernel_size, 2, cal_padding(x, kernel_size, 2))
        # The padding here may not be correct.

    def forward(self, x):
        return self.layer(x)

    def string(self):
        return self.name


class ResFcBlock(nn.Module):
    def __init__(self, x, out_dim, name, depth=2):
        self.name = name
        self.depth = depth
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(x.shape[-1], out_dim))
        self.shortcut = nn.Linear(x.shape[-1], out_dim)
    
    def forward(self, x):
        y = x
        for i in range(self.depth):
            y = self.layers[i](y)
        return y + self.shortcut(y)

    def string(self):
        return self.name


class ScaleFcBlock(nn.Module):
    def __init__(self, x, out_dim, name, block_per_scale=1, depth_per_block=2):
        self.name = name
        self.block_per_scale = block_per_scale
        self.layers = nn.ModuleList()
        for i in range(block_per_scale):
            temp = ResFcBlock(x, out_dim, 'block'+str(i), depth_per_block)
            self.layers.append(x)
            x = temp(x)
    
    def forward(self, x):
        y = x
        for i in range(self.block_per_scale):
            y = self.layers[i](y)
        return y

    def string(self):
        return self.name