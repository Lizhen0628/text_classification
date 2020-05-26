# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 10:01 上午
# @Author  : lizhen
# @FileName: data_process.py
# @Description:
from torch.utils.data import Dataset


class WordEmbedding():
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vectors = None


class Example():
    def __init__(self, text, label):
        self.label_id_map = {}
        self.text = text
        self.label = label
        self.label_id = self.label_id_map[label]

        self.tokens = []
        self.tokens_ids = []


class NLPDataSet(Dataset):
    def __init__(self, data_dir, data_name, batch_first=False, data_split=[0.3, 0.2]):
        """

        :param data_dir:  数据集所在的文件夹路径
        :param data_name:  数据集文件名称
        :param batch_first:
        :param data_split: 验证集和测试集的划分，默认：[0.3,0.2]测试集占30%，验证集占20%。
        """
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_first = batch_first
        self.data_split = data_split
        self.data = self._load_dataset()

    def _load_dataset(self):
        """
        加载源数据
        :return:
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
