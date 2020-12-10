import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Iterable

def cal_padding(input_shape, kernel_size, stride=1):
    m = []
    s = []
    if isinstance(kernel_size, Iterable):
        m = kernel_size
    else:
        m.append(kernel_size)
        m.append(kernel_size)
    if isinstance(stride, Iterable):
        s = stride
    else:
        s.append(stride)
        s.append(stride)
    n = input_shape[-2:]
    return [(((-n[i]) % s[i] - 1) * s[i] + m[i] + 1) // 2 for i in range(2)]


class ResBlock(nn.Module):
    def __init__(self, input_shape, out_dim, name, depth=2, kernel_size=3):
        super(ResBlock, self).__init__()
        self.depth = depth
        self.name = name
        
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.conv2ds = nn.ModuleList()
        cur_shape = input_shape
        cur_shape[0] = 1
        for _ in range(depth):
            self.batch_norms.append(nn.BatchNorm2d(cur_shape[1]))
            self.relus.append(nn.ReLU())
            temp = nn.Conv2d(cur_shape[1], out_dim, kernel_size, padding=cal_padding(cur_shape, kernel_size))
            self.conv2ds.append(temp)
            cur_shape = temp(torch.zeros(cur_shape)).shape
        self.s = nn.Conv2d(input_shape[1], out_dim, kernel_size, padding=cal_padding(input_shape, kernel_size))

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
    def __init__(self, input_shape, out_dim, name, block_per_scale=1, depth_per_block=2, kernel_size=3):
        super(ScaleBlock, self).__init__()
        self.block_per_scale = block_per_scale
        self.ResBlocks = nn.ModuleList()
        cur_shape = input_shape
        cur_shape = [1, input_shape[1], input_shape[2], input_shape[3]]
        for i in range(block_per_scale):
            temp = ResBlock(cur_shape, out_dim, 'block'+str(i), depth_per_block, kernel_size)
            self.ResBlocks.append(temp)
            cur_shape = temp(torch.zeros(cur_shape)).shape

    def forward(self, x):
        y = x
        for i in range(self.block_per_scale):
            y = self.ResBlocks[i](y)
        return y

    def string(self):
        return self.name


class DownSample(nn.Module):
    def __init__(self, input_shape, out_dim, kernel_size, name):
        super(DownSample, self).__init__()
        self.name = name
        assert(len(input_shape) == 4)
        self.layer = nn.Conv2d(input_shape[1], out_dim, kernel_size, 2, cal_padding(input_shape, kernel_size, 2))

    def forward(self, x):
        return self.layer(x)

    def string(self):
        return self.name


class UpSample(nn.Module):
    def __init__(self, input_shape, out_dim, kernel_size, name):
        super(UpSample, self).__init__()
        self.name = name
        assert(len(input_shape) == 4)
        self.layer = nn.ConvTranspose2d(input_shape[1], out_dim, kernel_size, 2, cal_padding(input_shape, kernel_size, 2))
        # The padding here may not be correct.

    def forward(self, x):
        return self.layer(x)

    def string(self):
        return self.name


class ResFcBlock(nn.Module):
    def __init__(self, input_shape, out_dim, name, depth=2):
        super(ResFcBlock, self).__init__()
        self.name = name
        self.depth = depth
        self.layers = nn.ModuleList()
        cur_shape = [1, input_shape[1]]
        for _ in range(depth):
            self.layers.append(nn.ReLU())
            temp = nn.Linear(cur_shape[-1], out_dim)
            self.layers.append(temp)
            cur_shape = temp(torch.zeros(cur_shape)).shape
        self.shortcut = nn.Linear(input_shape[-1], out_dim)
    
    def forward(self, x):
        y = x
        for i in range(self.depth):
            y = self.layers[i](y)
        return y + self.shortcut(x)

    def string(self):
        return self.name


class ScaleFcBlock(nn.Module):
    def __init__(self, input_shape, out_dim, name, block_per_scale=1, depth_per_block=2):
        super(ScaleFcBlock, self).__init__()
        self.name = name
        self.block_per_scale = block_per_scale
        self.layers = nn.ModuleList()
        cur_shape = [1, input_shape[1]]
        for i in range(block_per_scale):
            temp = ResFcBlock(cur_shape, out_dim, 'block'+str(i), depth_per_block)
            self.layers.append(temp)
            cur_shape = temp(torch.zeros(cur_shape)).shape
    
    def forward(self, x):
        y = x
        for i in range(self.block_per_scale):
            y = self.layers[i](y)
        return y

    def string(self):
        return self.name