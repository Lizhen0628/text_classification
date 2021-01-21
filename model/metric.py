# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 3:10 下午
# @Author  : jeffery
# @FileName: metric.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr


def tensor_to_numpy(preds, labels):
    return preds.cpu().detach().numpy(),labels.cpu().detach().numpy() # 用于多标签类别分类

def accuracy(preds, labels):
    """
    每个标签都计算accuracy，然后求平均，不考虑数据均衡问题
    """
    preds, labels = tensor_to_numpy(preds, labels)
    acc = accuracy_score(labels, preds, normalize=True)
    return acc


def macro_precision(preds, labels):
    """
    每个标签都计算precision，然后求平均，不考虑数据均衡问题
    """
    preds, labels = tensor_to_numpy(preds, labels)
    precision = precision_score(labels, preds, average='macro')
    return precision


def macro_recall(preds, labels):
    """
    每个标签都计算recall，然后求平均，不考虑数据均衡问题
    """
    preds, labels = tensor_to_numpy(preds, labels)
    recall = recall_score(labels, preds, average='macro')
    return recall


def macro_f1(preds, labels):
    """
    每个标签都计算f1，然后求平均，不考虑数据均衡问题
    """
    preds, labels = tensor_to_numpy(preds, labels)
    f1 = f1_score(labels, preds, average='macro')
    return f1


def micro_precision(preds, labels):
    """
    计算全数据的precision
    """
    preds, labels = tensor_to_numpy(preds, labels)
    return precision_score(labels, preds, average='micro')


def micro_recall(preds, labels):
    """
    计算全数据的recall
    """
    preds, labels = tensor_to_numpy(preds, labels)
    return recall_score(labels, preds, average='micro')


def micro_f1(preds, labels):
    """
    计算全数据的f1
    """
    preds, labels = tensor_to_numpy(preds, labels)
    return f1_score(labels, preds, average='micro')



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # max_preds = preds.argmax(dim=-1)  # get the index of the max probability
    correct = preds.eq(y)
    return (correct.sum().cpu() / torch.FloatTensor([y.shape[0]])).item()
