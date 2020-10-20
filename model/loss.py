# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 3:10 下午
# @Author  : jeffery
# @FileName: loss.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import torch.nn.functional as F
import torch
from torch import nn


def ce_loss(output, target):

    return F.cross_entropy(output, target)


def binary_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target.float())

def cosine_similarity_loss(first_vector,second_vector,label):
    return F.mse_loss(nn.Identity()(F.cosine_similarity(first_vector,second_vector)),label.view(-1))



def triple_loss(anchor_vec, pos_vec, neg_vec,triplet_margin=5):
    distance_pos = F.pairwise_distance(anchor_vec, pos_vec,p=2)
    distance_neg = F.pairwise_distance(anchor_vec, neg_vec,p=2)

    return F.relu(distance_pos - distance_neg + triplet_margin).mean()