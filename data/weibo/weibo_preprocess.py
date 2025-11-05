# -*- coding: utf-8 -*-
# @Time    : 2020/10/14 7:35 下午
# @Author  : jeffery
# @FileName: weibo_preprocess.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:

import sys
import os

# 获取当前脚本 (.../data/weibo/weibo_preprocess.py) 的路径
SCRIPT_PATH = os.path.abspath(__file__)

# 获取项目根目录 (.../text_classification)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_PATH)))

# 把项目根目录强行加入到 Python 的搜索路径中
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
import random
import numpy as np
import jieba
from gensim.models import KeyedVectors
from utils import WordEmbedding
import pickle



def convert_to_jsonl(input_file: Path, out_file: Path):
    writer = out_file.open('w')
    data_pd = pd.read_csv(input_file)
    for idx, row in tqdm(data_pd.iterrows()):
        item = {
            'id': idx,
            'text': row['review'],
            'labels': [row['label']]
        }
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()


def generate_al_data(input_file: Path):
    """
    以主动学习的方式来训练模型，需要准备的数据：
    训练集：active_learning_data/weibo_senti_train.jsonl,  20k条样本
    验证集：active_learning_data/weibo_senti_valid.jsonl,  10k条样本
    查询集：active_learning_data/weibo_senti_query.jsonl,  80k条样本
    测试集：active_learning_data/weibo_senti_test.jsonl,  9988条样本
    """

    # 定义输出目录
    output_dir = Path('active_learning_data')
    # 创建这个目录，如果它已存在则什么也不做 (exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = []
    with input_file.open('r') as f:
        for line in f:
            all_data.append(line)

    random.shuffle(all_data)
    # data分成训练集，验证集，查询集和测试集
    train_data = all_data[:20000]
    valid_data = all_data[20000:30000]
    query_data = all_data[30000:100000]
    test_data = all_data[100000:]

    # 训练集
    train_writer = (output_dir / 'weibo_senti_train.jsonl').open('w')
    for item in train_data:
        train_writer.write(item)
    train_writer.close()
    print('train...done...')

    # 验证集
    valid_writer = (output_dir / 'weibo_senti_valid.jsonl').open('w')
    for item in valid_data:
        valid_writer.write(item)
    valid_writer.close()
    print('valid...done...')

    # 查询集
    query_writer = (output_dir / 'weibo_senti_query.jsonl').open('w')
    for item in query_data:
        query_writer.write(item)
    query_writer.close()
    print('query...done...')

    # 测试集
    test_writer = (output_dir / 'weibo_senti_test.jsonl').open('w')
    for item in test_data:
        test_writer.write(item)
    test_writer.close()
    print('test...done...')




def make_word_embedding(input_file: Path, word_embedding: str):
    # 加载word embedding——sgns的词嵌入Q矩阵
    # wv包含了(word:词嵌入向量)的键值对
    wv = KeyedVectors.load_word2vec_format(word_embedding, binary=False, encoding='utf-8', unicode_errors='ignore')
    word_set = set()
    # 按词分
    # 在wv这个巨大的look-up table里查找在weibo数据集出现的词的词向量
    with input_file.open('r') as f:
        for line in tqdm(f):
            json_line = json.loads(line)
            word_set = word_set.union(set(jieba.lcut(json_line['text'])))

    stoi = defaultdict(int)
    itos = defaultdict(str)
    vectors = []
    for idx, word in enumerate(word_set):
        if word in wv.key_to_index:
            stoi[word] = len(stoi)
            itos[len(itos)] = word
            vectors.append(wv.get_vector(word))
    word_embedding = WordEmbedding(stoi=stoi, itos=itos, vectors=vectors)

    # 按字分
    # 在wv这个巨大的look-up table里查找在weibo数据集出现的字的词向量
    char_set = set()
    with input_file.open('r') as f:
        for line in tqdm(f):
            json_line = json.loads(line)
            char_set = char_set.union(set(list(json_line['text'])))

    stoi = defaultdict(int)
    itos = defaultdict(str)
    vectors = []
    for idx, char in enumerate(char_set):
        if char in wv.key_to_index:
            stoi[char] = len(stoi)
            itos[len(itos)] = char
            vectors.append(wv.get_vector(char))

    char_embedding = WordEmbedding(stoi=stoi, itos=itos, vectors=vectors)

    # 将出现的词和字对应的词向量存储在缓存中
    cache_dir = Path('../word_embedding/.cache')
    word_pkl_path = cache_dir / 'weibo_word_embedding.pkl'
    char_pkl_path = cache_dir / 'weibo_char_embedding.pkl'

    # 2. *** THIS IS THE FIX ***
    #    Create the directory (and any parent folders) if it doesn't exist.
    print(f"Creating cache directory at: {cache_dir.resolve()}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. Now, safely open and write the files
    #    (Using 'with open()' is safer and automatically closes files)
    with word_pkl_path.open('wb') as f_word:
        pickle.dump(word_embedding, f_word)
        
    with char_pkl_path.open('wb') as f_char:
        pickle.dump(char_embedding, f_char)

    print(f"Successfully saved word and char embeddings to {cache_dir}")


if __name__ == '__main__':
    input_file = Path('weibo_senti_100k.csv')
    out_file = Path('weibo_senti_100k.jsonl')
    convert_to_jsonl(input_file, out_file)
    generate_al_data(Path('weibo_senti_100k.jsonl'))
    make_word_embedding(Path('weibo_senti_100k.jsonl'), '../word_embedding/sgns.weibo.word')
