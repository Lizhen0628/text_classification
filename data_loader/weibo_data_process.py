# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 10:01 上午
# @Author  : lizhen
# @FileName: weibo_data_process.py
# @Description:

from base.base_data_loader import NLPDataset,NLPDataLoader
import pandas as pd

class WeiboDataset(NLPDataset):
    def __init__(self,data_path,test=False,stop_words_path=None,batch_first=False
                 ,include_lengths=False):

        csv_data = pd.read_csv(data_path)
        print('read data from {}'.format(data_path))
        text_field = "review"
        label_field = "label"
        super().__init__(data=csv_data,text_field=text_field,label_field=label_field,test=test,stop_words_path=stop_words_path
        ,batch_first=batch_first,include_lengths=include_lengths)



class WeiboDataLoader(NLPDataLoader):
    def __init__(self,dataset,split_ratio=None, batch_size=64,  device=None,train=True, repeat=False, shuffle=None, sort=None,sort_within_batch=False,
                 use_pretrained_word_embedding=False, word_embedding_path=None,vocab_size=2000):
        """
        微博二分类数据集加载器
        :param dataset:
        :param split_ratio:
        :param batch_size:
        :param device:
        :param train: Whether the iterator represents a train set
        :param repeat:
        :param shuffle:
        :param sort:
        :param sort_within_batch:
        :param use_pretrained_word_embedding:
        :param word_embedding_path:
        :param vocab_size:
        """

        super(WeiboDataLoader,self).__init__(dataset,split_ratio=split_ratio, batch_size=batch_size, sort_key=lambda x:len(x.text), device=device,
                 train=train, repeat=repeat, shuffle=shuffle, sort=sort,sort_within_batch=sort_within_batch,
                use_pretrained_word_embedding=use_pretrained_word_embedding, word_embedding_path=word_embedding_path,
                 vocab_size=vocab_size)
