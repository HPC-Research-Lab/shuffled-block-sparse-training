import math
import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F
import random
import os
import numpy as np
from extract_blocks import *
import math

class SparseLinear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 device=None,
                 dtype=None):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(SparseLinear, self).__init__()

        self.in_channels = in_features
        self.out_channels = out_features
        self.device = device
        self.sparsity = 0.9
        self.weight_num_active = 0
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.register_buffer('weight_mask', torch.ones((out_features, in_features), dtype=torch.bool))
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.weight_num = self.weight.shape[0] * self.weight.shape[1]
        self.register_forward_hook(self.forward_hook)
        self.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.input = input[0].detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.dout = grad_out[0].detach()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def init_linear_weight_mask(self, sparsity, block_size):
        self.sparsity = sparsity
        if self.sparsity != 0:
            self.weight_num_active = int(self.weight_num * (1 - self.sparsity))
            values, _ = torch.abs(self.weight.view(-1)).topk(self.weight_num_active, dim=0, largest=True, sorted=True)
            threshold = values[self.weight_num_active - 1]
            self.weight_mask[torch.abs(self.weight) < threshold] = 0
            # if block_size > 0:
            #     print('linear weight:',self.weight.shape)
            #     self.nblocks = int(math.ceil((1-self.sparsity) * self.weight_num / (block_size * block_size)))
            #     self.block_row_idx, self.block_col_idx, self.origin_groups = extract_dense_hygraph(self, block_size, self.nblocks)

    def update_linear_weight_mask(self, iterations, alpha, T_end, block_size):
        if self.sparsity != 0:
            if block_size > 0:
                sparsity_decay = alpha / 2 * (1 + math.cos(iterations * math.pi / T_end))
                drop_blocks = self.nblocks * sparsity_decay
                drop_num = int(torch.sum(self.weight_mask) * sparsity_decay)
                remain_num = self.weight_num_active - drop_num
    
                # 2. drop k elements in matrix
                weight_with_mask = torch.mul(self.weight, self.weight_mask).detach()
                values, _ = torch.abs(weight_with_mask.view(-1)).topk(remain_num, dim=0, largest=True, sorted=True)
                threshold_drop = values[remain_num - 1]
                drop_mask = torch.logical_and((torch.abs(weight_with_mask) < threshold_drop), self.weight_mask)
                self.weight_mask[drop_mask] = 0

                #grow same number of blocks
                dout_estimate = self.dout.view(self.out_channels, -1)
                input_estimate = self.input.view(self.in_channels, -1)
                norm_x, idx_x = torch.linalg.vector_norm(dout_estimate, dim=(0)).sort(descending=True)
                norm_y, idx_y = torch.linalg.vector_norm(input_estimate, dim=(0)).sort(descending=True)
    
                grow_x = int(round(math.sqrt(idx_x.shape[0] * drop_blocks / idx_y.shape[0] )))
                grow_y = int(round(math.sqrt(idx_y.shape[0]  * drop_blocks / idx_x.shape[0])))
                self.weight_mask.view(self.out_channels, -1)[torch.meshgrid(idx_x[:block_size*grow_x], idx_y[:block_size*grow_y])] = 1
                self.weight.data.view(self.out_channels, -1)[torch.meshgrid(idx_x[:block_size*grow_x], idx_y[:block_size*grow_y])] = 0
                # there still exist some overlapped weights, we need add more blocks back to the weight.
                ii = 0
                jj = 0
                while torch.sum(weight_mask) < torch.sum(mask_before):
                    print(torch.sum(weight_mask), torch.sum(mask_before), ii, jj)
                    if torch.sum(norm_x[block_size*(grow_x+ii):block_size*(grow_x+ii+1)]) > torch.sum(norm_y[block_size*(grow_y+jj):block_size*(grow_y+jj+1)]):
                        ii += 1
                        self.weight_mask.view(self.out_channels, -1)[torch.meshgrid(idx_x[block_size*(grow_x+ii):block_size*(grow_x+ii+1)], idx_y[block_size*(grow_y+jj):block_size*(grow_y+jj+1)])] = 1
                        self.weight.data.view(self.out_channels, -1)[torch.meshgrid(idx_x[block_size*(grow_x+ii):block_size*(grow_x+ii+1)], idx_y[block_size*(grow_y+jj):block_size*(grow_y+jj+1)])] = 0
                    else:
                        jj += 1
                        self.weight_mask.view(self.out_channels, -1)[torch.meshgrid(idx_x[block_size*(grow_x+ii):block_size*(grow_x+ii+1)], idx_y[block_size*(grow_y+jj):block_size*(grow_y+jj+1)])] = 1
                        self.weight.data.view(self.out_channels, -1)[torch.meshgrid(idx_x[block_size*(grow_x+ii):block_size*(grow_x+ii+1)], idx_y[block_size*(grow_y+jj):block_size*(grow_y+jj+1)])] = 0

    def update_linear_weight_mask1(self, iterations, alpha, T_end, block_size):
        if self.sparsity != 0:
            # 1.calculate the number K for update
            sparsity_decay = alpha / 2 * (1 + math.cos(iterations * math.pi / T_end))
            drop_num = int(torch.sum(self.weight_mask) * sparsity_decay)
            remain_num = self.weight_num_active - drop_num

            # 2. drop k elements in matrix
            weight_with_mask = torch.mul(self.weight, self.weight_mask).detach()
            values, _ = torch.abs(weight_with_mask.view(-1)).topk(remain_num, dim=0, largest=True, sorted=True)
            threshold_drop = values[remain_num - 1]
            drop_mask = torch.logical_and((torch.abs(weight_with_mask) < threshold_drop), self.weight_mask)
            self.weight_mask[drop_mask] = 0

            # 3. grow k elements in matrix
            grad_without_mask = torch.mul(self.weight.grad, torch.logical_not(self.weight_mask)).detach()
            if drop_num > 0:
                values, _ = torch.abs(grad_without_mask.view(-1)).topk(drop_num, dim=0, largest=True, sorted=True)
                threshold_grow = values[drop_num - 1]
                if threshold_grow != 0:
                    grow_mask = torch.abs(grad_without_mask) >= threshold_grow
                else:
                    grow_mask = torch.abs(grad_without_mask) > threshold_grow
                    grow_mask_rest = torch.abs(grad_without_mask) == threshold_grow
                    grow_mask_rest_idx = torch.nonzero(grow_mask_rest.view(-1))[: drop_num - int(torch.sum(grow_mask))]
                    grow_mask = grow_mask.view(-1)
                    grow_mask[grow_mask_rest_idx] = 1
                    grow_mask = grow_mask.reshape(grad_without_mask.shape)
            elif drop_num == 0:
                values, _ = torch.abs(grad_without_mask.view(-1)).topk(1, dim=0, largest=True, sorted=True)
                threshold_grow = values[0]
                grow_mask = torch.abs(grad_without_mask) > threshold_grow
            self.weight_mask[grow_mask] = 1
            self.weight.data[grow_mask] = 0

    

    # forward pass
    def forward(self, input):  
        weight_orig = self.weight.detach()
        self.weight.data = torch.mul(weight_orig, self.weight_mask)
        out = F.linear(input, self.weight, self.bias)
        return out
