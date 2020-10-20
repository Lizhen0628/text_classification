# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 1:06 下午
# @Author  : jeffery
# @FileName: query_strategies.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import torch.nn.functional as F


def margin_sampling(top_n, logists, embeddings, idxs_unlabeled):
    preds = F.softmax(logists, dim=-1)
    preds_sorted, idxs = preds.sort(descending=True)  # 对类别进行排序
    U = preds_sorted[:, 0] - preds_sorted[:, 1]  # 计算概率最高的前两个类别之前的差值，
    residule_sorted, idx_sorted = U.sort()  # 按照差值升序排序，差值越小说明预测效果越差
    return idxs_unlabeled[idx_sorted[:top_n]]
