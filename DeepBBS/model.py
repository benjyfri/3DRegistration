#!/usr/bin/env python
# -*- coding: utf-8 -*-


import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import soft_BBS_loss_torch, guess_best_alpha_torch, cdist_torch
import time
from geo_util import *
import torch.autograd.profiler as profiler

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
# Part of the code is referred from: https://github.com/WangYueFt/dcp

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_average_FPFH(x, neighbors, batch_size, num_points, k=20):
    FPFH_feats = []
    for batch in range(batch_size):
        FPFH_in_batch = []
        for point in range(num_points):
            patch = neighbors[batch, :, point, :]
            centroid = x[batch, :, point]
            fpfh_feat = calculateFPFH(centroid=centroid, patch_points=patch, max_nn=k)
            FPFH_in_batch.append(fpfh_feat)
        FPFH_feats.append(np.array(FPFH_in_batch))
    FPFH_feats_tensor = torch.tensor(np.array(FPFH_feats))

    # Calculate the mean along the last dimension
    FPFH_feats_tensor = torch.mean(FPFH_feats_tensor, dim=-1)

    return FPFH_feats_tensor


def get_persistent_homology(x, neighbors, batch_size, num_points):
    persistent_images = []
    for batch in range(batch_size):
        persistent_images_in_batch = []
        for point in range(num_points):
            patch = neighbors[batch, :, point, :]
            centroid = x[batch, :, point]
            p_i = calculatePersistentHomology(centroid=centroid, patch_points=patch)
            persistent_images_in_batch.append(p_i)
        persistent_images.append(np.array(persistent_images_in_batch))
    persistent_images_tensor = torch.tensor(np.array(persistent_images))

    flattened_persistent_images_tensor = persistent_images_tensor.view(
        persistent_images_tensor.size(0), persistent_images_tensor.size(1), -1)

    return flattened_persistent_images_tensor


import torch
import torch.linalg as linalg

def getEigens(points_tensor, k):
    """
    Compute the covariance matrix for each point and its k nearest neighbors.

    Args:
    points_tensor (torch.Tensor): Input tensor of 3D points with shape (num_of_batches, 3, size_of_batch).
    k (int): Number of nearest neighbors to consider.

    Returns:
    eigenvalues (torch.Tensor): Tensor of eigenvalues with shape (num_of_batches, 3, size_of_batch, 3).
    eigenvectors (torch.Tensor): Tensor of eigenvectors with shape (num_of_batches, 3, size_of_batch, 3, 3).
    """
    batch_size, _, num_points = points_tensor.size()

    # Reshape the points tensor for broadcasting
    points_tensor = points_tensor.permute(0, 2, 1)  # Shape: (num_of_batches, size_of_batch, 3)

    # Calculate L2 distances between points
    delta = points_tensor.unsqueeze(2) - points_tensor.unsqueeze(1)  # Shape: (num_of_batches, size_of_batch, size_of_batch, 3)
    squared_distances = torch.sum(delta ** 2, dim=-1)  # Shape: (num_of_batches, size_of_batch, size_of_batch)

    # Find the indices of k nearest neighbors
    _, indices = torch.topk(squared_distances, k, largest=False, sorted=True)  # Shape: (num_of_batches, size_of_batch, k)

    # Gather neighbor points using advanced indexing
    neighbors = torch.gather(points_tensor.unsqueeze(2).expand(-1, -1, k, -1),  # Shape: (num_of_batches, size_of_batch, k, 3)
                             dim=1,
                             index=indices.unsqueeze(-1).expand(-1, -1, -1, 3))

    # Calculate centered neighbors and covariance matrices
    centered_neighbors = neighbors - points_tensor.unsqueeze(2).expand(-1, -1, k, -1)
    covariance_matrices = torch.matmul(centered_neighbors.permute(0, 1, 3, 2), centered_neighbors) / k

    # Compute eigenvalues and eigenvectors
    eigenvalues, _ = linalg.eig(covariance_matrices)

    # Cast eigenvalues and eigenvectors to float64
    eigenvalues = eigenvalues.to(torch.float64)

    # Assuming eigenvalues has shape (num_of_batches, size_of_batch, 3) and eigenvectors has shape (num_of_batches, size_of_batch, 3, 3)

    # Get the sorted indices
    sorted_indices = torch.argsort(eigenvalues, dim=-1, descending=True)

    # Use the sorted indices to sort eigenvalues and eigenvectors
    sorted_eigenvalues = torch.gather(eigenvalues, -1, sorted_indices)


    # Calculate the largest eigenvalue divided by the sum of eigenvalues
    largest_eigenvalue = sorted_eigenvalues[..., 0]
    sum_of_eigenvalues = torch.sum(sorted_eigenvalues, dim=-1)
    largest_eigenvalue_ratio = largest_eigenvalue / sum_of_eigenvalues
    largest_eigenvalue_ratio = largest_eigenvalue_ratio.unsqueeze(-1)
    all_values  = torch.cat((eigenvalues, largest_eigenvalue_ratio), dim=2)
    return all_values




def getDensity(points_tensor, radius=0.05):
    """
    Count the number of points within a specified radius from each point in a 3D tensor.

    Args:
    points_tensor (torch.Tensor): A tensor of shape (num_of_batches, 3, size_of_batch) containing 3D points.
    radius (float): The radius within which to count points.

    Returns:
    torch.Tensor: A tensor of shape (num_of_batches, size_of_batch) containing the count of points within the radius for each point.
    """

    # Calculate the pairwise Euclidean distances between all points in the tensor
    num_of_batches, _, size_of_batch = points_tensor.size()
    permuted = points_tensor.permute(0,2,1)
    pairwise_distances = torch.cdist(permuted,permuted)
    # Count the number of points within the specified radius for each point
    num_points_within_radius = torch.sum(pairwise_distances <= radius, dim=2)
    ratio = num_points_within_radius / size_of_batch

    return ratio


def getFpfh(points_tensor, k=20):
    num_of_batches, _, size_of_batch = points_tensor.size()
    device = o3d.core.Device("CPU:0")
    if torch.cuda.is_available():
        device = o3d.core.Device("CUDA:0")

    # Create a PointCloud
    pcd = o3d.t.geometry.PointCloud(device)
    fpfh_per_batch =[]
    for batch_num in range (num_of_batches):
        batch = (points_tensor.permute(0,2,1))[batch_num,:,:]
        pack = torch.utils.dlpack.to_dlpack(batch)
        # Set the positions using the NumPy array
        pcd.point.positions = o3d.core.Tensor.from_dlpack(pack)

        # Compute normals
        pcd.estimate_normals(max_nn=k)

        # Compute FPFH features
        fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(pcd, max_nn=k, radius=None)
        tens = torch.utils.dlpack.from_dlpack(fpfh.to_dlpack())
        fpfh_per_batch.append(tens)
    # full_numpy = np.array(fpfh_per_batch)
    full_tensor = torch.stack(fpfh_per_batch, dim=0)
    return full_tensor
def get_geo_feature(x, k=20):
    start_time = time.time()
    feat = get_graph_feature(x,k)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time get_graph_feature : {elapsed_time}")
    start_time = time.time()
    batch_size, _, num_points = x.size()
    neighbors = (feat[:,0:3,:,:])
    fpfh_tensor = getFpfh(x, k=k)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time get_average_FPFH NEW: {elapsed_time}")
    start_time = time.time()
    persistent_images_tensor = get_persistent_homology(x, neighbors, batch_size, num_points)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time get_persistent_homology: {elapsed_time}")
    start_time = time.time()
    eigen_tensor = getEigens(x, k=k)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time getEigens: {elapsed_time}")
    start_time = time.time()
    density_tensor =  getDensity( x )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time getDensity: {elapsed_time}")
    start_time = time.time()
    print("----------------")
    print("DEVICES")
    print(f"density: {density_tensor.unsqueeze(-1).get_device()}, eigen_tensor: {eigen_tensor.get_device()},persistent_images_tensor: {persistent_images_tensor.get_device()}, fpfh_tensor: {fpfh_tensor.get_device()}")
    concatenated_tensor = torch.cat((density_tensor.unsqueeze(-1), eigen_tensor, persistent_images_tensor, fpfh_tensor), dim=-1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time concatenated_tensor: {elapsed_time}")
    if torch.cuda.is_available():
        # Move the tensor to the GPU using the .to() method
        concatenated_tensor = concatenated_tensor.to('cuda')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time TOCUDA: {elapsed_time}")
    return concatenated_tensor
def get_graph_feature(x, k=20, large_k=None):
    # x = x.squeeze()
    if large_k is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        idx = knn(x, k=large_k)
        idx = idx[:, :, torch.randperm(large_k)[:k]]
    batch_size, num_points, _ = idx.size()
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        print("Start geo")
        start_time = time.time()
        added_embedding = get_geo_feature(x)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"elapsed time{elapsed_time}")
        print("end geo")
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        x = torch.cat((x, added_embedding.permute(0, 2, 1)), dim=1)
        target_dim = 1024

        # Calculate the amount of padding required
        padding = target_dim - x.size(1)

        # Pad the tensor along the second dimension (emb) with zeros
        padded_tensor = torch.transpose(x, 1, 2)
        padded_tensor = torch.nn.functional.pad(padded_tensor, (0, padding), mode='constant', value=0)
        padded_tensor = torch.transpose(padded_tensor, 1, 2)
        padded_tensor = padded_tensor.to(torch.float32)
        return padded_tensor



class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.alpha_factor = args.alpha_factor
        self.eps = args.eps
        self.T_net = nn.Sequential(nn.Linear(self.emb_dims, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1),
                                   nn.ReLU())

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]

        batch_size = src.size(0)
        iter=input[4]
        device = src.device

        t = self.alpha_factor*torch.tensor([guess_best_alpha_torch(src_embedding[i,:], dim_num=1024, transpose=True) for i in range(batch_size)], device=device)
        scores = torch.cat(
            [soft_BBS_loss_torch(src_embedding[i,:], tgt_embedding[i,:], t[i], points_dim=1024, return_mat=True, transpose=True).float().unsqueeze(0)
            for i in range(batch_size)], dim=0)
        scores_norm = scores / (scores.sum(dim=2, keepdim=True)+self.eps)
        src_corr = torch.matmul(tgt, scores_norm.float().transpose(2, 1).contiguous())
        src_tgt_euc_dist = cdist_torch(src, tgt, 3)
        T = torch.clamp(self.T_net(torch.abs(src_embedding.mean(dim=2) - tgt_embedding.mean(dim=2))), 0.01, 100).view(-1,1,1)
        T = T/2**(iter-1)
        gamma = (scores * torch.exp(-src_tgt_euc_dist / T)).sum(dim=2, keepdim=True).float().transpose(2,1)
        src_weighted_mean = (src * gamma).sum(dim=2, keepdim=True) / (gamma.sum(dim=2, keepdim=True)+self.eps)
        src_centered = src - src_weighted_mean

        src_corr_weighted_mean = (src_corr * gamma).sum(dim=2, keepdim=True) / (gamma.sum(dim=2, keepdim=True) + self.eps)
        src_corr_centered = src_corr - src_corr_weighted_mean

        H = torch.matmul(src_centered * gamma, src_corr_centered.transpose(2, 1).contiguous()) + self.eps*torch.diag(torch.tensor([1,2,3], device=device)).unsqueeze(0).repeat(batch_size,1,1)

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src_weighted_mean) + src_corr_weighted_mean
        return R, t.view(batch_size, 3), src_corr

class SVDHead_no_network(nn.Module):
    def __init__(self):
        super(SVDHead_no_network, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3).cuda(), requires_grad=False)
        self.reflect[2, 2] = -1
        self.alpha_factor = 4
        self.eps = 1e-12

    def forward(self, src, tgt, iter=None):
        batch_size = src.size(0)
        device = src.device
        t = self.alpha_factor * torch.tensor([guess_best_alpha_torch(src[i, :], dim_num=3, transpose=True) for i in range(batch_size)], device=device)
        scores = torch.cat([soft_BBS_loss_torch(src[i, :], tgt[i, :], t[i], points_dim=3, return_mat=True, transpose=True).float().unsqueeze(0) for i in range(batch_size)], dim=0)
        scores_norm = scores / (scores.sum(dim=2, keepdim=True) + self.eps)
        src_corr = torch.matmul(tgt, scores_norm.float().transpose(2, 1).contiguous())

        src_tgt_euc_dist = cdist_torch(src, tgt, 3)
        T = 1
        T = T / 2 ** (iter - 1)
        gamma = (scores * torch.exp(-src_tgt_euc_dist / T)).sum(dim=2, keepdim=True).float().transpose(2, 1)
        src_weighted_mean = (src * gamma).sum(dim=2, keepdim=True) / (gamma.sum(dim=2, keepdim=True) + self.eps)
        src_centered = src - src_weighted_mean

        src_corr_weighted_mean = (src_corr * gamma).sum(dim=2, keepdim=True) / (
                    gamma.sum(dim=2, keepdim=True) + self.eps)
        src_corr_centered = src_corr - src_corr_weighted_mean

        H = torch.matmul(src_centered * gamma, src_corr_centered.transpose(2, 1).contiguous()) + self.eps * torch.diag(
            torch.tensor([1, 2, 3], device=device)).unsqueeze(0).repeat(batch_size, 1, 1)

        R = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

        R = torch.stack(R, dim=0)
        t = torch.matmul(-R, src_weighted_mean) + src_corr_weighted_mean
        return R, t.view(batch_size, 3), src_corr

class DCP(nn.Module):
    def __init__(self, args):
        super(DCP, self).__init__()
        self.emb_dims = args.emb_dims

        self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        self.pointer = Transformer(args=args)
        self.head = SVDHead(args=args)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]

        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)
        iter=input[2]

        # ############ADDED CODE##############
        # #Add to current embedding the awesome features we have
        #
        # src_added_embedding = get_geo_feature(src)
        # tar_added_embedding = get_geo_feature(tgt)
        # src_embedding = torch.cat((src_embedding, src_added_embedding.permute(0, 2, 1)), dim=1)
        # tgt_embedding = torch.cat((tgt_embedding, tar_added_embedding.permute(0, 2, 1)), dim=1)
        #
        # ############ADDED CODE##############


        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        rotation_ab, translation_ab, src_corr = self.head(src_embedding, tgt_embedding, src, tgt, iter)

        return rotation_ab, translation_ab, src_corr