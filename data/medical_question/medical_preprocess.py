# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 9:33 上午
# @Author  : jeffery
# @FileName: medical_preprocess.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from gensim.models import KeyedVectors
from utils import WordEmbedding, add_pad_unk
from collections import defaultdict
import pickle
import jieba


def convert_to_jsonl(input_file: Path, outfile: Path):
    writer = outfile.open('w')
    data_pd = pd.read_csv(input_file)
    for idx, row in tqdm(data_pd.iterrows()):
        item = {
            'id': row['ID'],
            'text': row['Question Sentence'],
            'labels': [int(row['category_A']), int(row['category_B']), int(row['category_C']), int(row['category_D']),
                       int(row['category_E']), int(row['category_F'])]
        }
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')

    writer.close()


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
    add_pad_unk(stoi, itos, vectors, wv)
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
    add_pad_unk(stoi, itos, vectors, wv)
    for idx, char in enumerate(char_set):
        if char in wv.vocab:
            stoi[char] = len(stoi)
            itos[len(itos)] = char
            vectors.append(wv.get_vector(char))

    char_embedding = WordEmbedding(stoi=stoi, itos=itos, vectors=vectors)

    word_embedding_cache = Path('../word_embedding/.cache/medical_word_embedding.pkl').open('wb')
    char_embedding_cache = Path('../word_embedding/.cache/medical_char_embedding.pkl').open('wb')
    pickle.dump(word_embedding, word_embedding_cache)
    pickle.dump(char_embedding, char_embedding_cache)
    word_embedding_cache.close()
    char_embedding_cache.close()


def split_train_valid(input_file: Path):
    """
    5000条样本：4000条训练，1000条验证
    """
    train_file_writer = Path('train_valid/medical_train.jsonl').open('w')
    valid_file_writer = Path('train_valid/medical_valid.jsonl').open('w')
    with input_file.open('r') as f:
        for idx, line in enumerate(f):
            if idx < 4000:
                train_file_writer.write(line)
            else:
                valid_file_writer.write(line)

    train_file_writer.close()
    valid_file_writer.close()


if __name__ == '__main__':
    # input_file =Path('train.csv')
    # output_file = Path('medical_question.jsonl')
    # convert_to_jsonl(input_file,output_file)

    # 注：这里使用的是搜狗新闻训练出来的词向量，如果可以找到医疗词向量来进行替换是比较好的选择（如果有大量医疗文本，也可以自行训练）
    # make_word_embedding(Path('medical_question.jsonl'), '../word_embedding/sgns.sogou.word')

    split_train_valid(Path('medical_question.jsonl'))
