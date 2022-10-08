import math
import scipy as sp
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from utils import *
import random
import torchvision
import numpy
from extract_blocks import *
import math


# using in Linux
# from torch.nn.common_types import _size_2_t
# using in Windows
from typing import TypeVar, Union, Tuple
T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

class SparseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            net=0,
            block=0,
            conv=0,
            device=None,
            dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [int(kernel_size), int(kernel_size)]
        self.kernel_size_ = _pair(kernel_size)
        self.stride_ = _pair(stride)
        self.padding_ = _pair(padding)
        self.dilation_ = _pair(dilation)
        self.groups = groups
        self.bias = bias
        self.net = net
        self.block = block
        self.conv = conv
        self.sparsity = 0.9
        self.weight_num_active = 0
        self.grad_block = None
        self.device = device
        self.nblocks = 0

        self.weight = Parameter(torch.empty(
            (self.out_channels, self.in_channels // self.groups, *self.kernel_size_), **factory_kwargs))
        self.bias = Parameter(torch.empty(self.out_channels, **factory_kwargs))
        self.register_buffer('weight_mask', torch.ones(
            (self.out_channels, self.in_channels // self.groups, *self.kernel_size_), dtype=torch.bool, device=self.device))

        self.reset_parameters()
        self.weight_num = self.weight.shape[0] * self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]

        self.register_forward_hook(self.forward_hook)
        self.register_full_backward_hook(self.backward_hook)


    def forward_hook(self, module, input, output):
        self.input = input[0].detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.dout = grad_out[0].detach()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def init_conv_weight_mask(self, sparsity, block_size):
        self.sparsity = sparsity
        if self.sparsity != 0:
            self.weight_num_active = int(self.weight_num * (1 - self.sparsity))
            values, _ = torch.abs(self.weight.view(-1)).topk(self.weight_num_active, dim=0, largest=True, sorted=True)
            threshold = values[self.weight_num_active - 1]
            self.weight_mask[torch.abs(self.weight) < threshold] = 0
            if block_size > 0 and self.out_channels % block_size == 0 and self.in_channels >= block_size :
                self.nblocks = int(math.ceil((1-self.sparsity) * self.weight_num / (block_size * block_size)))
                self.block_row_idx, self.block_col_idx = extract_dense(self, block_size, self.nblocks)

    def update_conv_weight_mask(self, iterations, alpha, T_end, block_size):
        rigl_drop_num = 0
        if self.sparsity != 0 and self.nblocks>0:
            mask_before = self.weight_mask.clone()
            sparsity_decay = alpha / 2 * (1 + math.cos(iterations * math.pi / T_end))
            drop_num = int(torch.sum(self.weight_mask) * sparsity_decay)
            remain_num = self.weight_num_active - drop_num
            self.pre_mask = self.weight_mask.clone()
            weight_with_mask = torch.mul(self.weight.detach(), self.weight_mask)
            values, _ = torch.abs(weight_with_mask.view(-1)).topk(remain_num, dim=0, largest=True, sorted=True)
            threshold_drop = values[remain_num-1]
            drop_mask = torch.logical_and((torch.abs(weight_with_mask) < threshold_drop), self.weight_mask)
            drop_mask_rest = torch.logical_and((torch.abs(weight_with_mask) == threshold_drop), self.weight_mask)
            drop_mask_idx = torch.nonzero(drop_mask_rest.view(-1))[: int(torch.sum(self.weight_mask)) - remain_num - int(torch.sum(drop_mask))]
            drop_mask = drop_mask.view(-1)
            drop_mask[drop_mask_idx] = 1
            dropped_mask = drop_mask.view(self.out_channels, -1)
            for i in range(len(self.block_row_idx)):
                row = self.block_row_idx[i]
                col = self.block_col_idx[i]
                temp_drop = dropped_mask[torch.meshgrid(row, col, indexing='ij')]
                col_mask_count = torch.sum(temp_drop, dim=0)
                row_mask_count = torch.sum(temp_drop, dim=1)
                col_mask_count = col_mask_count<block_size/2
                row_mask_count = row_mask_count<block_size/2
                set_B = ~(row_mask_count.reshape(-1,1)*col_mask_count)
                dropped_mask[torch.meshgrid(row, col, indexing='ij')] = torch.logical_and(set_B, temp_drop)
            drop_mask = dropped_mask.reshape(weight_with_mask.shape)
            self.weight_mask[drop_mask] = 0
            drop_num = drop_mask.sum().item()
            grad_without_mask = torch.mul(self.weight.grad, torch.logical_not(self.weight_mask)).detach()
            if drop_num > 0:
                values, _ = torch.abs(grad_without_mask.view(-1)).topk(drop_num, dim=0, largest=True, sorted=True)
                threshold_grow = values[drop_num-1]
                grow_mask = torch.logical_and(torch.abs(grad_without_mask) > threshold_grow, torch.logical_not(self.weight_mask))
                grow_mask_rest = torch.logical_and(torch.abs(grad_without_mask) == threshold_grow, torch.logical_not(self.weight_mask))
                grow_mask_rest_idx = torch.nonzero(grow_mask_rest.view(-1))[: drop_num - int(torch.sum(grow_mask))]
                grow_mask = grow_mask.view(-1)
                grow_mask[grow_mask_rest_idx] = 1
                grow_mask = grow_mask.reshape(grad_without_mask.shape)
            elif drop_num == 0:
                grow_mask = torch.zeros_like(drop_mask)
            self.weight_mask[grow_mask] = 1
            self.weight.data[grow_mask] = 0
            if block_size > 0 and self.nblocks != 0 and drop_num > 0:
                self.block_row_idx, self.block_col_idx = extract_dense(self, block_size, self.nblocks, grow_mask=grow_mask)

    def forward(self, input):
        self.weight.data = torch.mul(self.weight.detach(), self.weight_mask)

        out = F.conv2d(input, self.weight, self.bias, self.stride_,
                       self.padding_, self.dilation_, self.groups)
        return out