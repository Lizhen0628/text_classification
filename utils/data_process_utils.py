# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 9:37 上午
# @Author  : lizhen
# @FileName: data_process_utils.py
# @Description:
import os
import pickle
import numpy as np
from gensim.models import KeyedVectors


def load_pretrained_wordembedding(word_embedding_path):
    """
    加载预训练的词向量，并添加 'PAD'，'UNK' 以及生成对应的随机向量
    :return:
    """
    if not os.path.exists(word_embedding_path + '.pkl'):
        wv_from_text = KeyedVectors.load_word2vec_format(word_embedding_path, binary=False, encoding='utf-8',
                                                         unicode_errors='ignore')
        with open(word_embedding_path + '.pkl', 'wb') as f:
            pickle.dump(wv_from_text, f)
    else:
        with open(word_embedding_path + '.pkl', 'rb') as f:
            wv_from_text = pickle.load(f)
    wv_from_text.add('PAD', np.random.randn(wv_from_text.vector_size))
    wv_from_text.add('UNK', np.random.randn(wv_from_text.vector_size))

    return wv_from_text