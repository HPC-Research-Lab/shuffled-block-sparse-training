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
                #print(self.nblocks, drop_blocks)

                grow_x = int(round(math.sqrt(self.out_channels * drop_blocks / self.in_channels )))
                grow_y = int(round(math.sqrt(self.in_channels  * drop_blocks / self.out_channels)))

                drop_blocks = grow_x * grow_y

                #print(self.nblocks, drop_blocks)

                if drop_blocks > 0:
                    # drop some blocks
                    block_weight_with_mask = [torch.norm(self.weight.view(self.out_channels, -1)[torch.meshgrid(self.block_row_idx[i], self.block_col_idx[i], indexing='ij')]) ** 2 for i in range(self.nblocks)]
                    _, sorted_block_idx = torch.tensor(block_weight_with_mask).sort()
                    dropped_row = []
                    dropped_col = []
                    for i in range(drop_blocks):
                        r = self.block_row_idx[sorted_block_idx[i]]
                        c = self.block_col_idx[sorted_block_idx[i]]
                        dropped_row.append(r)
                        dropped_col.append(c)
                        self.weight_mask.view(self.out_channels, -1)[torch.meshgrid(r, c, indexing='ij')] = 0

                    self.after_dropped_block_row_idx = torch.tensor([self.block_row_idx[i].tolist() for i in sorted_block_idx[drop_blocks:]]).to(self.weight.device)
                    self.after_dropped_block_col_idx = torch.tensor([self.block_col_idx[i].tolist() for i in sorted_block_idx[drop_blocks:]]).to(self.weight.device)

                    #grow same number of blocks
                    _, idx_x = torch.linalg.vector_norm(self.dout, dim=(0)).sort(descending=True)
                    # _, idx_y = torch.linalg.vector_norm(input_window, dim=(0,2,3)).sort(descending=True)
                    _, idx_y = torch.linalg.vector_norm(self.input, dim=(0)).sort(descending=True)
                    
                    # dout:128(batch)*100(ic), input:128(batch)*128(oc)
                    assert(len(idx_x) >= grow_x * block_size)
                    assert(len(idx_y) >= grow_y * block_size)

                    R = idx_x[:grow_x*block_size]
                    C = idx_y[:grow_y*block_size]
                    self.grow_blocks = 0
                    self.drop_blocks = drop_blocks
                    self.merge_add_blocks(R, C, block_size, grow_y)

                    # print('first step: magnitude order------>grow blocks:', self.grow_blocks, 'drop blocks:', drop_blocks, 'sparsity:', self.weight_mask.sum().item()/self.weight.numel())

                    if self.grow_blocks< drop_blocks:
                        # print((grow_y+1)*block_size, len(idx_y))
                        if (grow_y+1)*block_size<len(idx_y):
                            new_C = idx_y[grow_y*block_size:(grow_y+1)*block_size]
                            self.merge_add_blocks(R, new_C, block_size,grow_y)
                            # print('second step: add col blocks------>grow blocks:', self.grow_blocks, 'drop blocks:', drop_blocks, 'sparsity:', self.weight_mask.sum().item()/self.weight.numel())
                    if self.grow_blocks<drop_blocks:
                        # print((grow_x+1)*block_size, len(idx_x))
                        if (grow_x+1)*block_size < len(idx_x) and (grow_y+1)*block_size<len(idx_y):
                            new_R = idx_x[grow_x*block_size:(grow_x+1)*block_size]
                            new_C = idx_y[:(grow_y+1)*block_size]
                            self.merge_add_blocks(new_R, new_C, block_size, grow_y)
                            # print('second step: add row blocks------>grow blocks:', self.grow_blocks, 'drop blocks:', drop_blocks, 'sparsity:', self.weight_mask.sum().item()/self.weight.numel())

                    if self.grow_blocks < drop_blocks:
                        while self.grow_blocks < drop_blocks:
                            temp_row = dropped_row[-1]
                            temp_col = dropped_col[-1]
                            temp_row = torch.tensor(temp_row).view(1,len(temp_row))
                            temp_col = torch.tensor(temp_col).view(1,len(temp_col))
                            self.after_dropped_block_row_idx = torch.cat((self.after_dropped_block_row_idx, temp_row),0)
                            self.after_dropped_block_col_idx = torch.cat((self.after_dropped_block_col_idx, temp_col),0)
                            grow_mask = torch.meshgrid(dropped_row[-1], dropped_col[-1], indexing='ij')
                            self.weight_mask.view(self.out_channels, -1)[grow_mask] = 1
                            # fix a bug
                            # self.weight.data.view(self.out_channels, -1)[grow_mask] = 0
                            dropped_row.pop()
                            dropped_col.pop()
                            self.grow_blocks+=1
                        # print('after', grow_blocks, drop_blocks)
                        # print('forth step: add back------>grow blocks:', self.grow_blocks, 'dropped blocks:', drop_blocks, 'sparsity:', self.weight_mask.sum().item()/self.weight.numel())
                    self.block_row_idx = self.after_dropped_block_row_idx
                    self.block_col_idx = self.after_dropped_block_col_idx

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
            # self.drop_random(drop_num)

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
