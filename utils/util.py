import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import jieba
import re
from tqdm import tqdm
import os
import numpy as np



def build_vocab(text, tokenizer, max_size, min_freq):
    UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
    vocab_dic = {}

    for line in tqdm(text):
        lin = clean_line(line).strip()
        if not lin:
            continue
        content = lin.split('\t')[0]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


# def load_dataset(text, vocab,tokenizer,pad_size=32):
#     UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
#     contents = []
#     text['review'] = text['review'].apply(tokenizer)
#
#     def cut_pad_sentence(token):
#         if len(token) < pad_size:
#             token.extend([PAD] * (pad_size - len(token)))
#             seq_len = len(token)
#         else:
#             token = token[:pad_size]
#             seq_len = pad_size
#
#         # word to id
#         for word in token:
#             words_line.append(vocab.get(word, vocab.get(UNK)))
#         contents.append((words_line, int(label), seq_len))
#     return contents  # [([...], 0), ([...], 1), ...]



def trimme_embedding(word_to_id,emb_dim,pretrain_embedding_dir,embedding_trimmed_dir):


    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_embedding_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    if not os.path.exists(embedding_trimmed_dir[:embedding_trimmed_dir.rindex('/')]):
        os.mkdir(embedding_trimmed_dir[:embedding_trimmed_dir.rindex('/')])
    np.savez_compressed(embedding_trimmed_dir, embeddings=embeddings)

def tokenizer(text):  # create a tokenizer function
    """
    定义分词操作
    """

    return list(jieba.cut(clean_line(text)))


# 分词，去停用词
def read_stop_words(filepath):
    """加载停用词文件"""
    return [line.strip() for line in open(filepath, 'r').readlines()]


def clean_line(s):
    """
    :param s: 清洗中文语料格式
    :return:
    """
    rule = re.compile(u'[^a-zA-Z0-9\u4e00-\u9fa5"#$%&\'()*+,-.:;<=>@\\^_`{|}]+')
    s = re.sub(rule, '', s)
    s = re.sub('[、]+', '，', s)
    s = re.sub('\'', '', s)
    s = re.sub('[#]+', '，', s)
    s = re.sub('[?]+', '？', s)
    s = re.sub('[;]+', '，', s)
    s = re.sub('[,]+', '，', s)
    s = re.sub('[!]+', '！', s)
    s = re.sub('[.]+', '.', s)
    s = re.sub('[，]+', '，', s)
    s = re.sub('[。]+', '。', s)
    s = re.sub('[~]+', '~', s)
    assert not len(s) == 0,'sentence length is zero'
    return s


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
