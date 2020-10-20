# -*- coding: utf-8 -*-
# @Time    : 2020/10/19 11:25 上午
# @Author  : jeffery
# @FileName: cnews_preprocess.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from pathlib import Path
from tqdm import tqdm
import json
from gensim.models import KeyedVectors
import jieba
from collections import defaultdict
from utils import WordEmbedding, add_pad_unk
import pickle


def convert_to_jsonl(raw_data_dir: Path, out_file: Path):
    text_length = []
    writer = out_file.open('w')

    idx = 0
    for txt in ['cnews.train.txt', 'cnews.val.txt', 'cnews.test.txt']:
        txt_file = raw_data_dir / '{}'.format(txt)
        with txt_file.open('r') as f:
            for line in tqdm(f):
                label, text = line.split('\t', 1)
                writer.write(json.dumps({
                    'id': idx,
                    'text': text,
                    'labels': [label]
                }, ensure_ascii=False) + '\n')
                idx += 1
                text_length.append(len(text))
    writer.close()
    print('sentence length : min:{},max:{},avg:{}'.format(min(text_length), max(text_length),
                                                          sum(text_length) / len(text_length)))
    print('sample num:{}'.format(idx))


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

    word_embedding_cache = Path('../word_embedding/.cache/cnews_word_embedding.pkl').open('wb')
    char_embedding_cache = Path('../word_embedding/.cache/cnews_char_embedding.pkl').open('wb')
    pickle.dump(word_embedding, word_embedding_cache)
    pickle.dump(char_embedding, char_embedding_cache)
    word_embedding_cache.close()
    char_embedding_cache.close()


def split_data(input_file: Path):
    train_file_writer = Path('train_valid_test/cnews_train.jsonl').open('w')
    valid_file_writer = Path('train_valid_test/cnews_valid.jsonl').open('w')
    test_file_writer = Path('train_valid_test/cnews_test.jsonl').open('w')
    with input_file.open('r') as f:
        for idx, line in enumerate(f):
            if idx < 50000:
                train_file_writer.write(line)
            if 50000 <= idx < 55000:
                valid_file_writer.write(line)
            if 55000 <= idx:
                test_file_writer.write(line)
    train_file_writer.close()
    valid_file_writer.close()
    test_file_writer.close()


if __name__ == '__main__':
    # raw_data_dir = Path('raw_data/')
    # out_file = Path('cnews_all.jsonl')
    # convert_to_jsonl(raw_data_dir, out_file)
    # input_file = Path('cnews_all.jsonl')
    # word_embedding = '../word_embedding/sgns.sogou.word'
    # make_word_embedding(input_file,word_embedding)
    split_data(Path('cnews_all.jsonl'))
