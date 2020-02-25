import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import jieba
import re


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
