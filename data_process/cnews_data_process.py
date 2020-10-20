# -*- coding: utf-8 -*-
# @Time    : 2020/10/19 2:08 下午
# @Author  : jeffery
# @FileName: cnews_data_process.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from base import BaseDataSet
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import pickle
import torch
import numpy as np
from tqdm import tqdm
import json
import jieba


@dataclass
class InputExample:
    guid: Optional[str]
    text: str
    label: int


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    guid: Optional[str]
    input_ids: List[int]
    attention_mask: Optional[List[int]]
    label: int

    def __post_init__(self):
        self.sent_len = len(self.input_ids)


class CnewsEmbeddingDataset(BaseDataSet):
    """
    该类用于非transformers，词嵌入使用wordembedding.
    """

    def __init__(self, data_dir, file_name, cache_dir, shuffle, word_embedding, overwrite_cache, batch_size,
                 num_workers):
        self.label_map_id = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
        self.data_dir = Path(data_dir)
        self.file_name = file_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.feature_cache_file = Path(cache_dir) / (file_name.split('.')[0] + '.cache')
        super().__init__(transformer_model=None, overwrite_cache=overwrite_cache,
                         force_download=None, cache_dir=None)

    def read_examples_from_file(self):
        input_file = self.data_dir / self.file_name
        with input_file.open('r') as f:
            for line in tqdm(f):
                json_line = json.loads(line)
                if len(json_line['text']) <= 7:
                    continue
                yield InputExample(guid=json_line['id'], text=json_line['text'],
                                   label=self.label_map_id[json_line['labels'][0]])

    def convert_examples_to_features(self):
        features = []
        for example in self.read_examples_from_file():
            tokens = jieba.lcut(example.text)
            input_ids = [
                self.word_embedding.stoi[token] if token in self.word_embedding.stoi else self.word_embedding.stoi[
                    'UNK'] for token in tokens]
            label = example.label
            features.append(InputFeatures(guid=example.guid, input_ids=input_ids, attention_mask=None, label=label))
        return features

    def collate_fn(self, datas):
        max_len = max([data.sent_len for data in datas])
        input_ids = []
        text_lengths = []
        labels = []

        for data in datas:
            input_ids.append(data.input_ids + [self.word_embedding.stoi['PAD']] * (max_len - data.sent_len))
            text_lengths.append(data.sent_len)
            labels.append(data.label)

        input_ids = torch.LongTensor(np.asarray(input_ids))
        text_lengths = torch.LongTensor(np.asarray(text_lengths))
        labels = torch.LongTensor(np.asarray(labels))
        return input_ids, None, text_lengths, labels  # 添加None是为了与使用transformer的模型对齐。


class CnewsTransformersDataset(BaseDataSet):

    def __init__(self, data_dir, file_name, cache_dir, shuffle, transformer_model, overwrite_cache, batch_size,
                 force_download,
                 num_workers):
        self.label_map_id = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
        self.data_dir = Path(data_dir)
        self.file_name = file_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_cache_file = Path(cache_dir) / (file_name.split('.')[0] + '.cache')
        super().__init__(transformer_model=transformer_model, overwrite_cache=overwrite_cache,
                         force_download=force_download, cache_dir=cache_dir)

    def read_examples_from_file(self):
        """
        注意这里限制了文本长度为：510，而cnews的平均长度为910。
        """
        input_file = self.data_dir / self.file_name
        with input_file.open('r') as f:
            for line in tqdm(f):
                json_line = json.loads(line)
                yield InputExample(guid=json_line['id'], text=json_line['text'][:510], label=self.label_map_id[json_line['labels'][0]])

    def convert_examples_to_features(self):
        features = []
        for example in self.read_examples_from_file():
            inputs = self.tokenizer.encode_plus(list(example.text), return_token_type_ids=False,
                                                return_attention_mask=True)
            features.append(InputFeatures(guid=example.guid, input_ids=inputs.data['input_ids'],
                                          attention_mask=inputs.data['attention_mask'], label=example.label))
        return features

    def collate_fn(self, datas):
        """
        dataloader collate function
        """
        max_len = max([data.sent_len for data in datas])
        input_ids = []
        attention_masks = []
        text_lengths = []
        labels = []

        for data in datas:
            input_ids.append(data.input_ids + [self.tokenizer.pad_token_id] * (max_len - data.sent_len))
            attention_masks.append(data.attention_mask + [0] * (max_len - data.sent_len))
            text_lengths.append(data.sent_len)
            labels.append(data.label)

        input_ids = torch.LongTensor(input_ids)
        attention_masks = torch.LongTensor(attention_masks)
        text_lengths = torch.LongTensor(text_lengths)
        labels = torch.LongTensor(labels)

        return input_ids,attention_masks,text_lengths,labels
