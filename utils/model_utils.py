# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 9:37 上午
# @Author  : lizhen
# @FileName: model_utils.py
# @Description:

import torch


def matrix_mul(input, weight, bias=False):
    """
    for HAN model
    :param input:
    :param weight:
    :param bias:
    :return:
    """
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, dim=0).squeeze(-1)
def element_wise_mul(input1, input2):
    """
    for HAN model
    :param input1:
    :param input2:
    :return:
    """
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 1)



def prepare_pack_padded_sequence( inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices