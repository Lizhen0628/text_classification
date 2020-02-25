# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 4:47 下午
# @Author  : lizhen
# @FileName: model.py
# @Description:
import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel



class RnnModel(BaseModel):
    def __init__(self, rnn_type, vocab, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_first=False,use_pretrain_embedding=False):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        if use_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.pad_index)
        # 把unknown 和 pad 向量设置为零
        self.embedding.weight.data[vocab.unk_index] = torch.zeros(embedding_dim)
        self.embedding.weight.data[vocab.pad_index] = torch.zeros(embedding_dim)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc = nn.Linear(hidden_dim * n_layers, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)

        if not self.bidirectional:
            hidden = torch.reshape(hidden,(hidden.shape[1],self.hidden_dim * self.n_layers))
        else:
            hidden = torch.reshape(hidden, (-1,hidden.shape[1], self.hidden_dim * self.n_layers))
            hidden = torch.mean(hidden,dim=0)
        output = torch.sum(output,dim=0)
        fc_input = self.dropout(output+hidden)
        out = self.fc(fc_input)

        return out






class FastText(BaseModel):
    def __init__(self, vocab, embedding_dim, output_dim, use_pretrain_embedding=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        if use_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.pad_index)
        # 把unknown 和 pad 向量设置为零
        self.embedding.weight.data[vocab.unk_index] = torch.zeros(embedding_dim)
        self.embedding.weight.data[vocab.pad_index] = torch.zeros(embedding_dim)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text,text_lengths):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        embedded = embedded.permute(1, 0, 2)

        # embedded = [batch size, sent len, emb dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # pooled = [batch size, embedding_dim]

        return self.fc(pooled)



class TextCNN(nn.Module):
    def __init__(self, vocab, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, use_pretrain_embedding=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        if use_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.pad_index)
        # 把unknown 和 pad 向量设置为零
        self.embedding.weight.data[vocab.unk_index] = torch.zeros(embedding_dim)
        self.embedding.weight.data[vocab.pad_index] = torch.zeros(embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text,text_lengths):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class TextCNN1d(nn.Module):
    def __init__(self, vocab, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, use_pretrain_embedding=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        if use_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.pad_index)
        # 把unknown 和 pad 向量设置为零
        self.embedding.weight.data[vocab.unk_index] = torch.zeros(embedding_dim)
        self.embedding.weight.data[vocab.pad_index] = torch.zeros(embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text,text_lengths):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

