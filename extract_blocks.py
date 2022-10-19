import torch
import os
import time
from sklearn.cluster import AgglomerativeClustering
from torch.nn import functional as F
from collections import Counter
import numpy as np


def extract_dense(layer, block_size, num_blocks, grow_mask=None):
	sparse_weight_mask = layer.weight_mask.view(layer.out_channels, -1)
	masked_weight = (torch.abs(layer.weight).detach() * layer.weight_mask).view(layer.out_channels, -1)

	if grow_mask != None:
		sparse_grow_mask = grow_mask.view(layer.out_channels, -1)
		# masked_weight[sparse_grow_mask==1] = torch.mean(masked_weight[masked_weight>0])
		masked_weight[sparse_grow_mask==1] = torch.quantile(masked_weight[masked_weight>0], 0.8,dim=0,keepdim=True,interpolation='higher')
	sim = torch.cosine_similarity(sparse_weight_mask.float().unsqueeze(1),sparse_weight_mask.float().unsqueeze(0), dim=-1)
	jaccard_similarity = 1 - sim
	cluster = AgglomerativeClustering(n_clusters=max(layer.out_channels//(block_size), 1), affinity='precomputed', linkage='complete')
	yy = cluster.fit(jaccard_similarity.cpu())
	z = yy.labels_
	clusters = {}
	for i in range(layer.out_channels):
		if z[i] not in clusters.keys():
			clusters[z[i]] = [i]
		else:
			clusters[z[i]].append(i)

	reordered = [x for k, v in sorted(clusters.items(), key=lambda t: len(t[1]), reverse=True) for x in v]
	groups = torch.split(torch.tensor(reordered), block_size)
	block_row_indices = torch.tensor([], dtype=torch.int).to(layer.weight.device)
	block_col_indices = torch.tensor([], dtype=torch.int).to(layer.weight.device)
	block_w = torch.tensor([]).to(layer.weight.device)


	for g in groups:
		g = g.to(layer.weight.device)
		val = torch.linalg.norm(masked_weight[g], dim=0)
		w, idx = val.sort(descending=True)
		w_blocked = torch.stack(torch.split(w, block_size))
		idx_blocked = torch.stack(torch.split(idx, block_size))
		t = torch.sum(w_blocked, dim=1)
		block_w = torch.cat((block_w, t[t>0]), 0)
		block_row_indices = torch.cat((block_row_indices, g.view(-1,block_size).repeat(len(t[t>0]),1)), 0)
		block_col_indices = torch.cat((block_col_indices, idx_blocked[t>0]), 0)

	_, sorted_idx = block_w.sort(descending=True)

	block_row_indices = torch.tensor([block_row_indices[i].tolist() for i in sorted_idx[:num_blocks]]).to(layer.weight.device)
	block_col_indices = torch.tensor([block_col_indices[i].tolist() for i in sorted_idx[:num_blocks]]).to(layer.weight.device)

	layer.weight_mask.zero_()
	for i in range(num_blocks):
		r = block_row_indices[i]
		c = block_col_indices[i]
		layer.weight_mask.view(layer.out_channels, -1)[torch.meshgrid(r, c, indexing='ij')] = 1

	return block_row_indices, block_col_indices
