# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 10:01 上午
# @Author  : lizhen
# @FileName: weibo_data_process.py
# @Description:

from base.base_data_loader import NLPDataset, NLPDataLoader
import pandas as pd
from utils.util import read_stop_words, tokenizer
from tqdm import tqdm
import torch
from torchtext.data import Dataset, LabelField, Field, Example
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
from pytorch_transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np
import pandas as pd


class WeiboDataset(NLPDataset):
    def __init__(self, data_path, test=False, stop_words_path=None, bert_model_path=None, batch_first=False,
                 include_lengths=False, tokenizer_language='cn'):
        """
        :param data_path:
        :param test: 如果为测试集，则不加载label
        :param stop_words_path:
        :param batch_first:
        :param include_lengths:
        """
        self.data = pd.read_csv(data_path)
        print('read data from {}'.format(data_path))
        self.text_field = "review"
        self.label_field = "label"
        self.test = test

        if stop_words_path:
            stop_words = read_stop_words(stop_words_path)
        else:
            stop_words = None

        self.LABEL = LabelField(sequential=False, use_vocab=False, dtype=torch.float)

        # lambda x: [y for y in x]
        # bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        # pad_index = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.pad_token)
        # unk_index = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.unk_token)
        self.TEXT = Field(use_vocab=True, sequential=True, stop_words=stop_words, tokenize=tokenizer,
                          batch_first=batch_first,
                          tokenizer_language=tokenizer_language,
                          include_lengths=include_lengths)  # include_lengths=True for LSTM

        self.fields = [("text", self.TEXT), ("label", self.LABEL)]

        self.examples = self.build_examples()

    def build_examples(self):
        examples = []
        if self.test:
            # 如果为测试集，则不加载label
            for text in tqdm(self.data[self.text_field]):
                examples.append(Example.fromlist([text, None], self.fields))
        else:
            for text, label in tqdm(zip(self.data[self.text_field], self.data[self.label_field])):
                # Example: Defines a single training or test example.
                # Stores each column of the example as an attribute.
                examples.append(Example.fromlist([text, label], self.fields))
        return examples


class BertDataLoader(NLPDataLoader):
    def __init__(self, dataset, split_ratio=None, batch_size=64, sort_key=lambda x: len(x.text), device=None,
                 train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=False,
                 batch_size_fn=None, use_pretrained_word_embedding=False, word_embedding_path=None):
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
        self.word_embedding_path = word_embedding_path
        self.train_iter, self.valid_iter, self.test_iter = self.get_iterators()

    def get_iterators(self):
        train_iter, valid_iter, test_iter = None, None, None
        if not self.split_ratio:
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


class WeiboDataLoader(NLPDataLoader):
    def __init__(self, dataset, split_ratio=None, batch_size=64, sort_key=lambda x: len(x.text), device=None,
                 train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=False,
                 batch_size_fn=None, use_pretrained_word_embedding=False, word_embedding_path=None):
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
        self.word_embedding_path = word_embedding_path
        self.train_iter, self.valid_iter, self.test_iter = self.get_iterators()
        self.vocab = self.dataset.TEXT.vocab
        self.vocab.pad_index = self.dataset.TEXT.vocab.stoi[self.dataset.TEXT.pad_token]

    def get_iterators(self):
        train_iter, valid_iter, test_iter = None, None, None
        if not self.split_ratio:
            # 构建词汇表
            if self.use_pretrained_word_embedding:
                vectors = Vectors(name=self.word_embedding_path)  # 使用预训练的词向量
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
                vectors = Vectors(name=self.word_embedding_path)  # 使用预训练的词向量
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


class BertDataProcess():
    def __init__(self, data_path, batch_size, device, processed_data_path=None, bert_model_path=None):

        self.bert_model_path = bert_model_path
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.CLS = self.bert_tokenizer.cls_token
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device
        self.processed_data_path = processed_data_path  # 经过load_data 处理过的数据保存到该路径下。
        self.train_data,self.val_data = self.built_dataset()
        self.train_iter = self.built_iterater(self.train_data)
        self.val_iter = self.built_iterater(self.val_data)

    def load_dataset(self, path):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if 'label,review' in lin:
                    continue
                if not lin:
                    continue

                label, content = lin.split(',',1)
                # print("content>>>:", content)
                token = self.bert_tokenizer.tokenize(content)
                token = [self.CLS] + token
                seq_len = len(token)
                token_ids = self.bert_tokenizer.convert_tokens_to_ids(token)
                contents.append((token_ids, int(label), seq_len))
        return contents

    def built_dataset(self):

        if not os.path.exists(self.processed_data_path):
            all_data = self.load_dataset(self.data_path)
            pickle.dump(all_data, open(self.processed_data_path,'wb'))
        else:
            all_data = pickle.load(open(self.processed_data_path,'rb'))

        train_data, val_data = train_test_split(all_data, test_size=0.3, shuffle=True)

        return train_data, val_data

    def built_iterater(self, dataset):
        return BertDatasetIterater(dataset, self.batch_size, self.device,self.bert_model_path)



class BertDatasetIterater(object):
    def __init__(self, batches, batch_size, device,bert_model_path):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
    def _mask_sentence_len(self,content,max_sent_len):
        pad_index = self.bert_tokenizer.pad_token_id
        sent_len = len(content)
        mask = [1]*sent_len + ([0] * (max_sent_len-sent_len))
        content = content.extend([pad_index] * (max_sent_len-sent_len))
        return mask



    def _to_tensor(self, datas):
        datas = pd.DataFrame(datas,columns=['content','label','sent len'])
        max_sent_len = max(datas['sent len'])
        datas["mask"] = datas["content"].apply(self._mask_sentence_len,args={max_sent_len})

        x = torch.LongTensor(pd.DataFrame(datas['content'].to_list()).values).to(self.device)
        y = torch.FloatTensor(datas['label'].values).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor(datas['sent len']).to(self.device)
        mask = torch.LongTensor(pd.DataFrame(datas['mask'].to_list()).values).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches
