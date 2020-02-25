# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 10:01 上午
# @Author  : lizhen
# @FileName: weibo_data_process.py
# @Description:
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
import pandas as pd
from torchtext.data import Dataset, LabelField, Field, Example
from utils.util import tokenizer,clean_line,read_stop_words
from tqdm import tqdm
import torch
import os

class NLPDataLoader():
    def __init__(self, dataset,split_ratio=None, batch_size=64, sort_key=None, device=None,
                 batch_size_fn=None, train=True, repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None, use_pretrained_word_embedding=False,word_embedding_name=None, word_embedding_path=None):

        # 构建数据集
        self.dataset = dataset
        # 迭代器参数
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.device = device
        self.batch_size_fn = batch_size_fn
        self.train = train
        self.repeat = repeat
        self.shuffle = shuffle
        self.sort = sort
        self.sort_within_batch = sort_within_batch
        self.use_pretrained_word_embedding = use_pretrained_word_embedding
        self.word_embedding_name = word_embedding_name
        self.word_embedding_path = word_embedding_path
        self.train_iter,self.valid_iter,self.test_iter = self.get_iterators()
        self.pad_idx = self.dataset.TEXT.vocab.stoi[self.dataset.TEXT.pad_token]
        self.unk_idx = self.dataset.TEXT.vocab.stoi[self.dataset.TEXT.unk_token]
        self.vocab = self.dataset.TEXT.vocab

    def get_iterators(self):
        train_iter, valid_iter, test_iter = None, None, None
        if not self.split_ratio:
            # 构建词汇表
            if self.use_pretrained_word_embedding:
                vectors = Vectors(name=self.word_embedding_name,cache=self.word_embedding_path)  # 使用预训练的词向量
                self.dataset.TEXT.build_vocab(self.dataset, vectors=vectors,
                                              unk_init=torch.Tensor.normal_)
                self.dataset.LABEL.build_vocab(self.dataset)
                print('label types:{}'.format(self.dataset.LABEL.vocab.stoi))
            else:
                self.dataset.TEXT.build_vocab(self.dataset)  # 不使用预训练的词向量
                self.dataset.LABEL.build_vocab(self.dataset)
                print('label types:{}'.format(self.dataset.LABEL.vocab.stoi))

            train_iter = Iterator(self.dataset, batch_size=self.batch_size, device=self.device, sort_key=self.sort_key,
                                  sort_within_batch=self.sort_within_batch, repeat=self.repeat,
                                  batch_size_fn=self.batch_size_fn
                                  , train=self.train, shuffle=self.shuffle, sort=self.sort)

        else:
            train_data, valid_data, test_data = None, None, None
            if isinstance(self.split_ratio, list):
                if len(self.split_ratio) == 3:
                    train_data, valid_data, test_data = self.dataset.split(split_ratio=self.split_ratio)
                else:
                    train_data, valid_data = self.dataset.split(split_ratio=self.split_ratio)
            else:
                train_data, valid_data = self.dataset.split(split_ratio=self.split_ratio)

            # 构建词汇表
            if self.use_pretrained_word_embedding:
                vectors = Vectors(name=self.word_embedding_name,cache=self.word_embedding_path)  # 使用预训练的词向量
                self.dataset.TEXT.build_vocab(train_data, vectors=vectors,
                                              unk_init=torch.Tensor.normal_)
                self.dataset.LABEL.build_vocab(train_data)
                print('label types:{}'.format(self.dataset.LABEL.vocab.stoi))
            else:
                self.dataset.TEXT.build_vocab(train_data)  # 不使用预训练的词向量
                self.dataset.LABEL.build_vocab(train_data)
                print('label types:{}'.format(self.dataset.LABEL.vocab.stoi))

            # 构建迭代器

            if train_data:
                train_iter = Iterator(train_data, batch_size=self.batch_size, device=self.device,
                                      sort_key=self.sort_key,
                                      sort_within_batch=self.sort_within_batch, repeat=self.repeat,
                                      batch_size_fn=self.batch_size_fn
                                      , train=self.train, shuffle=self.shuffle, sort=self.sort)
            if valid_data:
                valid_iter = Iterator(valid_data, batch_size=self.batch_size, device=self.device,
                                      sort_key=self.sort_key,
                                      sort_within_batch=self.sort_within_batch, repeat=self.repeat,
                                      batch_size_fn=self.batch_size_fn
                                      , train=self.train, shuffle=self.shuffle, sort=self.sort)
            if test_data:
                test_iter = Iterator(test_data, batch_size=self.batch_size, device=self.device, sort_key=self.sort_key,
                                     sort_within_batch=self.sort_within_batch, repeat=self.repeat,
                                     batch_size_fn=self.batch_size_fn
                                     , train=self.train, shuffle=self.shuffle, sort=self.sort)

        return train_iter, valid_iter, test_iter



class NLPDataset(Dataset):
    def __init__(self, data, text_field, label_field, test=False,stop_words_path=None, batch_first=False
                 , include_lengths=False,tokenizer_language='cn',):
        if stop_words_path:
            stop_words = read_stop_words(stop_words_path)
        else:
            stop_words = None

        self.LABEL = LabelField(sequential=False, use_vocab=False, dtype=torch.float)


        self.TEXT = Field(sequential=True,stop_words=stop_words, tokenize=tokenizer, batch_first=batch_first,tokenizer_language=tokenizer_language,
                          include_lengths=include_lengths)  # include_lengths=True for LSTM

        fields = [("text", self.TEXT), ("label", self.LABEL)]

        examples = []
        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(data[text_field]):
                examples.append(Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(data[text_field], data[label_field])):
                # Example: Defines a single training or test example.
                # Stores each column of the example as an attribute.
                examples.append(Example.fromlist([text, label], fields))
        # 之前是一些预处理操作，此处调用super调用父类构造方法，产生标准Dataset
        super(NLPDataset, self).__init__(examples, fields)
