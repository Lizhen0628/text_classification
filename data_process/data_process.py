# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 10:01 上午
# @Author  : lizhen
# @FileName: data_process.py
# @Description:


from base.base_dataset import NLPDataSet, WordEmbedding
import pandas as pd
import numpy as np
from utils.data_process_utils import load_pretrained_wordembedding
from tqdm import tqdm
import jieba
from sklearn.model_selection import train_test_split
import pickle
import os
import torch
from transformers import BertTokenizer, XLNetTokenizer


class CnewsExample():
    def __init__(self, text, label):
        self.label_id_map = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
        self.text = text
        self.label = label
        self.label_id = self.label_id_map[label]

        self.tokens = []
        self.tokens_ids = []


class CnewsDataset(NLPDataSet):

    def __init__(self, data_dir, train_name, valid_name, test_name, word_embedding_path, device, max_sent_len,
                 xlnet_path=None, use_han=False):
        """

        :param data_dir:  数据集所在的文件夹路径
        :param data_name:  数据集文件名称
        :param batch_first:
        :param data_split: 验证集和测试集的划分，默认：[0.3,0.2]测试集占30%，验证集占20%。
        """
        # 构建样本数据集
        self.device = device
        self.data_dir = data_dir
        self.train_name = train_name
        self.valid_name = valid_name
        self.test_name = test_name
        self.word_embedding_path = word_embedding_path
        self.max_sent_len = max_sent_len

        if xlnet_path:
            # 加载bert分词器
            self.xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_path)
            self.train_set = self._load_dataset_4_xlnet(self.train_name)
            self.valid_set = self._load_dataset_4_xlnet(self.valid_name)
            self.test_set = self._load_dataset_4_xlnet(self.test_name)
            # 为了保持model输入的统一
            self.word_embedding = None
        elif use_han:
            self.train_set = self._load_dataset_4_han(self.train_name)
            self.valid_set = self._load_dataset_4_han(self.valid_name)
            self.test_set = self._load_dataset_4_han(self.test_name)
        else:
            self.train_set, self.word_embedding = self._load_dataset(self.train_name)
            self.valid_set = self._load_dataset(self.valid_name)
            self.test_set = self._load_dataset(self.test_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _load_dataset_4_xlnet(self, file_name):
        if not os.path.exists(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_xlnet.pkl')):
            examples = []

            with open(os.path.join(self.data_dir, file_name), 'r') as f:
                for line in tqdm(f):
                    label, text = line.split('\t')
                    if len(text) > self.max_sent_len:
                        text = text[:self.max_sent_len]
                    cnews_example = CnewsExample(text, label)
                    cnews_example.tokens = self.xlnet_tokenizer.tokenize(cnews_example.text)
                    cnews_example.tokens_ids = self.xlnet_tokenizer.encode(cnews_example.tokens,
                                                                           add_special_tokens=True)
                    examples.append(cnews_example)

            with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_xlnet.pkl'), 'wb') as f:
                pickle.dump(examples, f)
        else:
            with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_xlnet.pkl'), 'rb') as f:
                examples = pickle.load(f)

        return examples

    def _load_dataset_4_han(self, file_name):


        if not os.path.exists(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_han.pkl')):
            examples = []
            # 加载训练集上的word embedding
            with open(os.path.join(self.data_dir, 'word_embedding.pkl'), 'rb') as f:
                self.word_embedding = pickle.load(f)
            if '[SEP]' not in self.word_embedding.stoi:
                self.word_embedding.stoi['[SEP]'] = len(self.word_embedding.stoi)
                self.word_embedding.itos[len(self.word_embedding.stoi)] = '[SEP]'
            with open(os.path.join(self.data_dir, file_name), 'r') as f:
                for line in tqdm(f):
                    label, text = line.split('\t')
                    # if len(text) > self.max_sent_len:
                    #     text = text[:self.max_sent_len]
                    cnews_example = CnewsExample(text, label)
                    # 使用字向量
                    # cnews_example.tokens = list(cnews_example.text)
                    # 使用词向量
                    cnews_example.tokens = [*jieba.lcut(cnews_example.text)]
                    for token in cnews_example.tokens:
                        if token == '。':
                            token = '[SEP]'
                        if token in self.word_embedding.stoi:
                            cnews_example.tokens_ids.append(self.word_embedding.stoi[token])
                        else:
                            cnews_example.tokens_ids.append(self.word_embedding.stoi['UNK'])
                    examples.append(cnews_example)
            # 保存成pkl文件，方便加载
            with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_han.pkl'), 'wb') as f:
                pickle.dump(examples, f)
        else:
            # 加载训练集上的word embedding
            with open(os.path.join(self.data_dir, 'word_embedding.pkl'), 'rb') as f:
                self.word_embedding = pickle.load(f)

            with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_han.pkl'), 'rb') as f:
                examples = pickle.load(f)
        return examples

    def _load_dataset(self, file_name):
        """
        加载数据集，并构建词汇表，word embedding
        :return:
        """

        # 只在训练集上构建word embedding
        if 'train' in file_name:
            # 如果pkl 文件不存在:1.加载训练集 2.构建训练集上的词汇表 3.构建训练集上的word embedding
            if not os.path.exists(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_char.pkl')):
                examples = []
                # 加载预训练的word embedding
                pretrained_wordembedding = load_pretrained_wordembedding(self.word_embedding_path)
                stoi = {}
                itos = {}
                stoi['UNK'] = 0
                stoi['PAD'] = 1
                itos[0] = 'UNK'
                itos[1] = 'PAD'
                # 根据训练数据构建word_embedding id:-1.08563066e+00  9.97345448e-01  2.82978505e-01 -1.50629473e+00........
                vectors = []
                vectors.append(pretrained_wordembedding['UNK'])
                vectors.append(pretrained_wordembedding['PAD'])
                with open(os.path.join(self.data_dir, file_name), 'r') as f:
                    for line in tqdm(f):
                        label, text = line.split('\t')
                        if len(text) > self.max_sent_len:
                            text = text[:self.max_sent_len]
                        cnews_example = CnewsExample(text, label)
                        # 使用字向量
                        # cnews_example.tokens = list(cnews_example.text)
                        # 使用词向量
                        cnews_example.tokens = [*jieba.lcut(cnews_example.text)]
                        for token in cnews_example.tokens:
                            if token in pretrained_wordembedding:
                                if token not in stoi:
                                    stoi[token] = len(stoi)
                                    itos[len(stoi)] = token
                                    vectors.append(pretrained_wordembedding[token])
                                cnews_example.tokens_ids.append(stoi[token])
                            else:
                                cnews_example.tokens_ids.append(stoi['UNK'])
                        examples.append(cnews_example)
                word_embedding = WordEmbedding(stoi, itos)
                word_embedding.vectors = np.array(vectors)

                # 保存成pkl文件，方便加载
                with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_char.pkl'), 'wb') as f:
                    pickle.dump(examples, f)
                with open(os.path.join(self.data_dir, 'word_embedding.pkl'), 'wb') as f:
                    pickle.dump(word_embedding, f)
            else:

                with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_char.pkl'), 'rb') as f:
                    examples = pickle.load(f)
                with open(os.path.join(self.data_dir, 'word_embedding.pkl'), 'rb') as f:
                    word_embedding = pickle.load(f)

            return examples, word_embedding
        else:
            if not os.path.exists(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_char.pkl')):
                examples = []
                # 加载训练集上的word embedding
                with open(os.path.join(self.data_dir, 'word_embedding.pkl'), 'rb') as f:
                    word_embedding = pickle.load(f)
                with open(os.path.join(self.data_dir, file_name), 'r') as f:
                    for line in tqdm(f):
                        label, text = line.split('\t')
                        if len(text) > self.max_sent_len:
                            text = text[:self.max_sent_len]
                        cnews_example = CnewsExample(text, label)
                        # 使用字向量
                        # cnews_example.tokens = list(cnews_example.text)
                        # 使用词向量
                        cnews_example.tokens = [*jieba.lcut(cnews_example.text)]
                        for token in cnews_example.tokens:
                            if token in word_embedding.stoi:
                                cnews_example.tokens_ids.append(word_embedding.stoi[token])
                            else:
                                cnews_example.tokens_ids.append(word_embedding.stoi['UNK'])
                        examples.append(cnews_example)
                # 保存成pkl文件，方便加载
                with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_char.pkl'), 'wb') as f:
                    pickle.dump(examples, f)
            else:
                with open(os.path.join(self.data_dir, file_name.rsplit('.', 1)[0] + '_char.pkl'), 'rb') as f:
                    examples = pickle.load(f)
            return examples

    def bert_collate_fn(self, datas):

        # 记录batch中每个句子长度
        seq_lens = []
        input_token_ids = []
        class_label = []
        bert_masks = []
        # 获取该batch 中句子的最大长度
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)

        # padding 句子到相同长度
        for data in datas:
            class_label.append(data.label_id)
            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.xlnet_tokenizer.pad_token_id] * (max_seq_len - cur_seq_len))
            bert_masks.append([1] * len(data.tokens_ids) + [0] * (max_seq_len - cur_seq_len))
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        bert_masks = torch.ByteTensor(np.array(bert_masks)).to(self.device)
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        class_label = torch.LongTensor(np.array(class_label)).to(self.device)

        return input_token_ids, bert_masks, seq_lens, class_label

    def collate_fn(self, datas):
        """
        训练阶段所使用的collate function

        class CnewsExample():
            def __init__(self, text, label):
                self.label_id_map = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
                self.text = text
                self.label = label
                self.label_id = self.label_id_map[label]

                self.tokens = []
                self.tokens_ids = []

        :return:
        """
        # 记录batch中每个句子长度
        seq_lens = []
        input_token_ids = []
        class_labels = []
        # 获取该batch 中句子的最大长度
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)

        # padding 句子到相同长度
        for data in datas:
            class_labels.append(data.label_id)
            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.word_embedding.stoi['PAD']] * (max_seq_len - cur_seq_len))

        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        class_labels = torch.LongTensor(np.array(class_labels)).to(self.device)

        return input_token_ids, None, seq_lens, class_labels


class WeiboExample():
    def __init__(self, text, label):
        self.text = text
        self.label = label

        self.tokens = []
        self.tokens_ids = []


class WeiboDataSet(NLPDataSet):

    def __init__(self, data_dir, data_name, word_embedding_path, device, test_size=0.3,
                 bert_path=None):
        """

        :param data_dir:  数据集所在的文件夹路径
        :param data_name:  数据集文件名称
        :param batch_first:
        :param data_split: 验证集和测试集的划分，默认：[0.3,0.2]测试集占30%，验证集占20%。
        """
        # 构建样本数据集
        self.device = device
        self.data_dir = data_dir
        self.data_name = data_name
        self.test_size = test_size
        self.word_embedding_path = word_embedding_path

        if bert_path:
            # 加载bert分词器
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            self.data = self._load_dataset_4_bert()
            # 为了保持model输入的统一
            self.word_embedding = None
        else:
            self.data, self.word_embedding = self._load_dataset()
        self.train_set, self.test_set = train_test_split(self.data, test_size=self.test_size)

    def _load_dataset_4_bert(self):
        """
        加载数据集，因为使用bert，分词略有不同。
        :return:
        """
        file_name = self.data_name.split('.')[0]
        if not os.path.exists(os.path.join(self.data_dir, file_name + '_bert.pkl')):
            examples = []

            raw_data = pd.read_csv(os.path.join(self.data_dir, self.data_name), header=0, names=['label', 'text'])
            for item in tqdm(raw_data.iterrows()):
                weibo_example = WeiboExample(item[1]['text'], item[1]['label'])
                weibo_example.tokens = self.bert_tokenizer.tokenize(weibo_example.text)
                weibo_example.tokens_ids = self.bert_tokenizer.encode(weibo_example.tokens, add_special_tokens=True)
                examples.append(weibo_example)

            with open(os.path.join(self.data_dir, file_name + '_bert.pkl'), 'wb') as f:
                pickle.dump(examples, f)
        else:
            with open(os.path.join(self.data_dir, file_name + '_bert.pkl'), 'rb') as f:
                examples = pickle.load(f)

        return examples

    def _load_dataset(self):
        """
        加载数据集，并构建词汇表，word embedding
        :return:
        """
        file_name = self.data_name.split('.')[0]
        # 如果pkl 文件不存在:1.加载训练集 2.构建训练集上的词汇表 3.构建训练集上的word embedding
        if not os.path.exists(os.path.join(self.data_dir, file_name + '.pkl')):
            pretrained_wordembedding = load_pretrained_wordembedding(self.word_embedding_path)  # 加载预训练的word embedding
            examples = []
            # 根据训练数据构建词汇表  'token':id
            stoi = {}
            itos = {}
            stoi['UNK'] = 0
            stoi['PAD'] = 1
            itos[0] = 'UNK'
            itos[1] = 'PAD'
            # 根据训练数据构建word_embedding id:-1.08563066e+00  9.97345448e-01  2.82978505e-01 -1.50629473e+00........
            vectors = []
            vectors.append(pretrained_wordembedding['UNK'])
            vectors.append(pretrained_wordembedding['PAD'])

            raw_data = pd.read_csv(os.path.join(self.data_dir, self.data_name), header=0, names=['label', 'text'])
            for item in tqdm(raw_data.iterrows()):
                weibo_example = WeiboExample(item[1]['text'], item[1]['label'])
                # 使用词向量
                weibo_example.tokens = [*jieba.lcut(weibo_example.text)]
                # 使用字向量
                # weibo_example.tokens = list(weibo_example.text)
                for token in weibo_example.tokens:

                    if token in pretrained_wordembedding:  # 如果token在预训练的word embedding 词汇表中
                        if token not in stoi:
                            stoi[token] = len(stoi)
                            itos[len(stoi)] = token
                            vectors.append(pretrained_wordembedding[token])

                        weibo_example.tokens_ids.append(stoi[token])
                    else:  # 如果token 不在预训练的word embedding 词汇表中
                        weibo_example.tokens_ids.append(stoi['UNK'])
                examples.append(weibo_example)

            word_embedding = WordEmbedding(stoi, itos)
            word_embedding.vectors = np.array(vectors)

            # 保存成pkl文件，方便加载
            with open(os.path.join(self.data_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(examples, f)
            with open(os.path.join(self.data_dir, 'word_embedding.pkl'), 'wb') as f:
                pickle.dump(word_embedding, f)
        else:
            with open(os.path.join(self.data_dir, file_name + '.pkl'), 'rb') as f:
                examples = pickle.load(f)
            with open(os.path.join(self.data_dir, 'word_embedding.pkl'), 'rb') as f:
                word_embedding = pickle.load(f)

        return examples, word_embedding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def bert_collate_fn(self, datas):

        # 记录batch中每个句子长度
        seq_lens = []
        input_token_ids = []
        class_label = []
        bert_masks = []
        # 获取该batch 中句子的最大长度
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)

        # padding 句子到相同长度
        for data in datas:
            class_label.append(data.label)
            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.bert_tokenizer.pad_token_id] * (max_seq_len - cur_seq_len))
            bert_masks.append([1] * len(data.tokens_ids) + [0] * (max_seq_len - cur_seq_len))
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        bert_masks = torch.ByteTensor(np.array(bert_masks)).to(self.device)
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)

        return input_token_ids, bert_masks, seq_lens, class_label

    def collate_fn(self, datas):
        """
        训练阶段所使用的collate function

        class WeiboExample():
            def __init__(self, text, label):
                self.text = text
                self.label = label

                self.tokens = []
                self.tokens_ids = []

        :return:
        """
        # 记录batch中每个句子长度
        seq_lens = []
        input_token_ids = []
        class_label = []
        # 获取该batch 中句子的最大长度
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)

        # padding 句子到相同长度
        for data in datas:
            class_label.append(data.label)
            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.word_embedding.stoi['PAD']] * (max_seq_len - cur_seq_len))

        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)

        return input_token_ids, None, seq_lens, class_label
