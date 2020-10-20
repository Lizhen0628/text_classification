# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 10:07 上午
# @Author  : jeffery
# @FileName: medical_question_data_process.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from base import BaseDataSet
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass
import torch
import numpy as np
import jieba
from tqdm import tqdm
import pickle
import json


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
    labels: List[int]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    guid: Optional[str]
    input_ids: List[int]
    attention_mask: Optional[List[int]]
    label: List[int]

    def __post_init__(self):
        self.sent_len = len(self.input_ids)


class MedicalEmbeddingDataset(BaseDataSet):
    """
    该类用于非transformers，词嵌入使用wordembedding.
    """

    def __init__(self, data_dir, file_name, cache_dir, shuffle, word_embedding, overwrite_cache, batch_size,
                 num_workers):
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
                yield InputExample(guid=json_line['id'], text=json_line['text'], labels=json_line['labels'])

    def convert_examples_to_features(self):
        features = []
        for example in self.read_examples_from_file():
            tokens = jieba.lcut(example.text)
            input_ids = [
                self.word_embedding.stoi[token] if token in self.word_embedding.stoi else self.word_embedding.stoi[
                    'UNK'] for token in tokens]
            features.append(
                InputFeatures(guid=example.guid, input_ids=input_ids, attention_mask=None, label=example.labels))
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


class MedicalTransformersDataset(BaseDataSet):

    def __init__(self, data_dir, file_name, cache_dir, shuffle, transformer_model, overwrite_cache, batch_size,
                 force_download,
                 num_workers):
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
                yield InputExample(guid=json_line['id'], text=json_line['text'][:510], labels=json_line['labels'])

    def convert_examples_to_features(self):
        features = []
        for example in self.read_examples_from_file():
            inputs = self.tokenizer.encode_plus(list(example.text), return_token_type_ids=False,
                                                return_attention_mask=True)
            features.append(InputFeatures(guid=example.guid, input_ids=inputs.data['input_ids'],
                                          attention_mask=inputs.data['attention_mask'], label=example.labels))
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

        return input_ids, attention_masks, text_lengths, labels
