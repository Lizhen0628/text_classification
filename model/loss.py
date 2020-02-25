# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 5:35 下午
# @Author  : lizhen
# @FileName: loss.py
# @Description:
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


def nll_loss(output, target):
    return F.nll_loss(output, target)


def crossentropy_loss(output, target):
    return F.cross_entropy(output, target)


def binary_crossentropy_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
