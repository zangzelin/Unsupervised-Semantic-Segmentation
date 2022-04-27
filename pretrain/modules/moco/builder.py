#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#
# MoCo implementation based upon https://arxiv.org/abs/1911.05722
# 
# Pixel-wise contrastive loss based upon our paper


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common_config import get_model
from modules.losses import BalancedCrossEntropyLoss
import scipy
import numpy as np

def UMAPNoSigmaSimilarity(dist, gamma, v=100, h=1, pow=2):

    dist_rho = dist

    dist_rho[dist_rho < 0] = 0
    Pij = (
        gamma
        * torch.tensor(2 * 3.14)
        * gamma
        * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
    )
    return Pij

def CalGamma(v):
    from mpmath import gamma #print(float(gamma(0.5))) 
    a = float(gamma((v + 1) / 2))
    b = np.sqrt(v * np.pi) * float(gamma(v / 2)) 
    out = a / b

    return out

class ContrastiveModel(nn.Module):
    def __init__(self, p):
        """
        p: configuration dict
        """
        super(ContrastiveModel, self).__init__()

        self.K = p['moco_kwargs']['K'] 
        self.m = p['moco_kwargs']['m'] 
        self.T = p['moco_kwargs']['T']

        # create the model 
        self.model_q = get_model(p)
        self.model_k = get_model(p)

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.dim = p['model_kwargs']['ndim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # self.lc = nn.Sequential(
        #     nn.Linear(self.dim, self.dim),
        #     nn.Linear(self.dim, self.dim),
        # )

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # balanced cross-entropy loss
        self.bce = BalancedCrossEntropyLoss(size_average=True)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def pdist2(self, x: torch.Tensor, y: torch.Tensor):
        # calculate the pairwise distance

        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)
        return dist

    def forward(self, im_q, im_k, sal_q, sal_k):
        """
        Input:
            images: a batch of images (B x 3 x H x W) 
            sal: a batch of saliency masks (B x H x W)
        Output:
            logits, targets
        """
        batch_size = im_q.size(0)
        self.dim_lat = 256

        q_lat, q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        # print('q_lat.shape', q_lat.shape)
        # print('q.shape', q.shape)
        q = nn.functional.normalize(q, dim=1)
        q = q.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # queries: pixels x dim

        q_lat = nn.functional.normalize(q_lat, dim=1)
        q_lat = q_lat.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        q_lat = torch.reshape(q_lat, [-1, self.dim_lat]) # queries: pixels x dim

        pixels = q_lat.shape[0]

        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)
   


        # compute key prototypes
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_lat, k, _ = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)
            k_lat = nn.functional.normalize(k_lat, dim=1)
            # print('k_lat.shape, k.shape', k_lat.shape, k.shape)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_lat = self._batch_unshuffle_ddp(k_lat, idx_unshuffle)
            # print('k_lat.shape, k.shape', k_lat.shape, k.shape)
            
            # prototypes k
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            k_lat = k_lat.reshape(batch_size, self.dim_lat, -1) # B x dim x H.W
            # print('k.shape, k_lat.shape', k.shape, k_lat.shape)
            
            sal_k_1 = sal_k.reshape(batch_size, -1, 1).type(k.dtype) # B x H.W x 1
            sal_k_lat = sal_k.reshape(batch_size, -1, 1).type(k_lat.dtype) # B x H.W x 1
            # print('sal_k_1.shape, sal_k_lat.shape', sal_k_1.shape, sal_k_lat.shape)
            
            prototypes_foreground = torch.bmm(k, sal_k_1).squeeze() # B x dim
            prototypes_foreground_lat = torch.bmm(k_lat, sal_k_lat).squeeze() # B x dim
            
            
            prototypes = nn.functional.normalize(prototypes_foreground, dim=1)        
            prototypes_lat = nn.functional.normalize(prototypes_foreground_lat, dim=1)    

        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            sal_q = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            sal_q = sal_q.view(-1)
            mask_indexes = torch.nonzero((sal_q)).view(-1).squeeze()
            sal_q = torch.index_select(sal_q, index=mask_indexes, dim=0) // 2

            posi = F.one_hot(sal_q)
            # nega = torch.zeros((mask_indexes.shape[0], self.queue.shape[1])).to(sal_q.device)
            # sal_q_new = torch.cat([posi, nega], dim=1)

            # print('posi.shape', posi.shape)
            # print('nega.shape', nega.shape)
            # pixels x (proto + negatives)
            # print('sal_q_new.shape', sal_q_new.shape)

        # q: pixels x dim
        # k: pixels x dim
        # prototypes_k: proto x dim
        q = torch.index_select(q, index=mask_indexes, dim=0)
        q_lat = torch.index_select(q_lat, index=mask_indexes, dim=0)
        
        l_batch = torch.matmul(q, prototypes.t())   # shape: pixels x proto
        # print('q_lat.shape', q_lat.shape)
        # print('prototypes_lat.shape', prototypes_lat.shape)
        # print('q.shape', q.shape)
        # print('prototypes.shape', prototypes.shape)
        d_1 = self.pdist2(q_lat, prototypes_lat)
        d_2 = self.pdist2(q, prototypes)

        v_lat = 100.0
        v_emb = 1.0
        d_1[posi==1] = d_1[posi==1] / 10
        P = UMAPNoSigmaSimilarity(d_1, CalGamma(v_lat), v=v_lat)
        Q = UMAPNoSigmaSimilarity(d_2, CalGamma(v_emb), v=v_emb)

        # negatives = self.queue.clone().detach()     # shape: dim x negatives
        # l_mem = torch.matmul(q, negatives)          # shape: pixels x negatives (Memory bank)
        # logits = torch.cat([l_batch, l_mem], dim=1) # pixels x (proto + negatives)


        # apply temperature
        # logits /= self.T
        # logits = logits-logits.min()
        # logits = logits/logits.max()

        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes) 

        # print('logits.shape', logits.shape)
        
        # print('logits.max()', logits.max())
        # print('logits.min()', logits.min())


        return Q, sal_q, P, sal_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
