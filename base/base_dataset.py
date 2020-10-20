# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 5:03 下午
# @Author  : jeffery
# @FileName: base_dataset.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from transformers import LongformerTokenizer, BertTokenizer, AlbertTokenizer, AutoTokenizer,AutoConfig
from pathlib import Path
# import tfrecord
import random
from dataclasses import dataclass
from typing import List, Optional, Union
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset
import torch

@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: Optional[str]
    text: str
    label: int


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    # token_type_ids: Optional[List[int]] = None
    label: int

    def __post_init__(self):
        self.sent_len = len(self.input_ids)

class BaseDataSet(Dataset):
    """
    适合内存可以加载全部数据的情况
    """

    def __init__(self, transformer_model, overwrite_cache,force_download,cache_dir):
        # 分词器
        if transformer_model:
            self.transformer_config = AutoConfig.from_pretrained(transformer_model,force_download=force_download,cache_dir=cache_dir)
            # self.tokenizer = AutoTokenizer.from_pretrained(transformer_model,force_download=force_download,cache_dir=cache_dir)
            self.tokenizer = BertTokenizer.from_pretrained(transformer_model,force_download=force_download,cache_dir=cache_dir) # clue/albert_chinese_tiny
        if not self.feature_cache_file.exists() or overwrite_cache:
            self.features = self.save_features_to_cache()
        else:
            self.features = self.load_features_from_cache()

    def read_examples_from_file(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def save_features_to_cache(self):
        features = self.convert_examples_to_features()
        if self.shuffle:
            random.shuffle(features)
        print('saving feature to cache file : {}...'.format(self.feature_cache_file))
        torch.save(features, self.feature_cache_file)
        return features

    def load_features_from_cache(self):
        print('loading features from cache file : {}...'.format(self.feature_cache_file))
        return torch.load(self.feature_cache_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def make_k_fold_data(self, k):
        """
        :param k:
        :return:
        """
        kf = KFold(n_splits=k)  # 分成几个组
        index_collecter = []
        for train_index, valid_index in kf.split(self.features):
            index_collecter.append((train_index, valid_index))
        return index_collecter


class BaseTFRecordDataSet:
    """
        适合内存无法一次直接加载的情况，可以处理一批数据，然后写入一批数据,数据以tfrecord的格式存储
    """

    def __init__(self, data_dir, shuffle, pretrained_dir, data_mode):
        # 分词器
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

        # 数据模式：训练集/测试集/验证集
        self.data_dir = Path(data_dir)
        self.data_mode = data_mode
        self.examples_tfrecord = self.data_dir / '{}.tfrecord'.format(data_mode)
        self.shuffle = shuffle

    def _make_tfrecord(self):

        if not self.examples_tfrecord.exists():
            self.examples_tfrecord.touch()

        writer = tfrecord.TFRecordWriter(self.examples_tfrecord)
        examples = self.get_examples()
        if self.shuffle:
            random.shuffle(examples)
        for e in examples:
            writer.write(e)
        writer.close()

    def get_examples(self):
        raise NotImplementedError

    def _load_tfrecord(self):
        # index_path =None
        # description = {
        #     "diag_desc": "byte",
        #     "code": "byte",
        #     "code_id": "int",
        #     "label": "float",
        #     "code_desc": "byte",
        #     "input_ids": "int",
        #     "attention_mask": "int",
        #     "token_type_ids": "int"
        # }
        # dataset = TFRecordDataset(self.examples_tfrecord,index_path,description,)

        # return dataset
        raise NotImplementedError
