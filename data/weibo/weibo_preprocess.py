# -*- coding: utf-8 -*-
# @Time    : 2020/10/14 7:35 下午
# @Author  : jeffery
# @FileName: weibo_preprocess.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
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
    all_data = []
    with input_file.open('r') as f:
        for line in f:
            all_data.append(line)

    random.shuffle(all_data)
    train_data = all_data[:100]
    valid_data = all_data[100:11100]
    query_data = all_data[11100:101000]
    test_data = all_data[101000:]

    # 训练集
    train_writer = Path('active_learning_data/weibo_senti_train.jsonl').open('w')
    for item in train_data:
        train_writer.write(item)
    train_writer.close()
    print('train...done...')

    # 验证集
    valid_writer = Path('active_learning_data/weibo_senti_valid.jsonl').open('w')
    for item in valid_data:
        valid_writer.write(item)
    valid_writer.close()
    print('valid...done...')

    # 查询集
    query_writer = Path('active_learning_data/weibo_senti_query.jsonl').open('w')
    for item in query_data:
        query_writer.write(item)
    query_writer.close()
    print('query...done...')

    # 测试集
    test_writer = Path('active_learning_data/weibo_senti_test.jsonl').open('w')
    for item in test_data:
        test_writer.write(item)
    test_writer.close()
    print('test...done...')




def make_word_embedding(input_file: Path, word_embedding: str):
    # 加载word embedding
    wv = KeyedVectors.load_word2vec_format(word_embedding, binary=False, encoding='utf-8', unicode_errors='ignore')

    word_set = set()
    # 按词分
    with input_file.open('r') as f:
        for line in tqdm(f):
            json_line = json.loads(line)
            word_set = word_set.union(set(jieba.lcut(json_line['text'])))

    stoi = defaultdict(int)
    itos = defaultdict(str)
    vectors = []
    for idx, word in enumerate(word_set):
        if word in wv.vocab:
            stoi[word] = len(stoi)
            itos[len(itos)] = word
            vectors.append(wv.get_vector(word))
    word_embedding = WordEmbedding(stoi=stoi, itos=itos, vectors=vectors)

    # 按字分
    char_set = set()
    with input_file.open('r') as f:
        for line in tqdm(f):
            json_line = json.loads(line)
            char_set = char_set.union(set(list(json_line['text'])))

    stoi = defaultdict(int)
    itos = defaultdict(str)
    vectors = []
    for idx, char in enumerate(char_set):
        if char in wv.vocab:
            stoi[char] = len(stoi)
            itos[len(itos)] = char
            vectors.append(wv.get_vector(char))

    char_embedding = WordEmbedding(stoi=stoi, itos=itos, vectors=vectors)

    word_embedding_cache = Path('../word_embedding/.cache/weibo_word_embedding.pkl').open('wb')
    char_embedding_cache = Path('../word_embedding/.cache/weibo_char_embedding.pkl').open('wb')
    pickle.dump(word_embedding, word_embedding_cache)
    pickle.dump(char_embedding, char_embedding_cache)
    word_embedding_cache.close()
    char_embedding_cache.close()


if __name__ == '__main__':
    # input_file = Path('weibo_senti_100k.csv')
    # out_file = Path('weibo_senti_100k.jsonl')
    # convert_to_jsonl(input_file, out_file)
    generate_al_data(Path('weibo_senti_100k.jsonl'))
    # make_word_embedding(Path('weibo_senti_100k.jsonl'), '../word_embedding/sgns.weibo.word')
