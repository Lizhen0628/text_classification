# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 4:47 下午
# @Author  : lizhen
# @FileName: model.py
# @Description:
import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from operator import itemgetter
from transformers import BertModel, XLNetModel
from utils.model_utils import prepare_pack_padded_sequence, matrix_mul, element_wise_mul
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np


class DPCNN(nn.Module):
    def __init__(self, num_filters, num_classes, word_embedding, freeze):
        super(DPCNN, self).__init__()

        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        self.conv_region = nn.Conv2d(1, num_filters, (3, self.embedding_size), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, text, _, text_lengths):
        # text [batch_size,seq_len]
        x = self.embedding(text)  # x=[batch_size,seq_len,embedding_dim]
        x = x.unsqueeze(1).to(torch.float32)  # [batch_size, 1, seq_len, embedding_dim]
        x = self.conv_region(x)  # x = [batch_size, num_filters, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)  # [batch_size, num_filters,1,1]
        x = x.squeeze()  # [batch_size, num_filters]
        x = self.fc(x)  # [batch_size, 1]
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


class RnnModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers, bidirectional, dropout, word_embedding, freeze,
                 batch_first=True):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc = nn.Linear(hidden_dim * n_layers, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, _, text_lengths):
        # 按照句子长度从大到小排序
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text = [batch size,sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            # output (seq_len, batch, num_directions * hidden_size)
            # hidden (num_layers * num_directions, batch, hidden_size)
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # 把句子序列再调整成输入时的顺序
        output = output[desorted_indices]
        # output = [batch_size,seq_len,hidden_dim * num_directionns ]
        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.mean(output, dim=1)
        fc_input = self.dropout(output + hidden)
        out = self.fc(fc_input)

        return out


class RCNNModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers, bidirectional, dropout, word_embedding, freeze,
                 batch_first=True):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc_cat = nn.Linear(hidden_dim * n_layers + self.embedding_size, self.embedding_size)
        self.fc = nn.Linear(self.embedding_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, _, text_lengths):
        # 按照句子长度从大到小排序
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text = [batch size,sent len]
        embedded = self.dropout(self.embedding(text)).to(torch.float32)
        # embedded = [batch size, sent len, emb dim]
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)

        # packed_output
        # hidden [n_layers * bi_direction,batch_size,hidden_dim]
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output [sent len, batch_size * n_layers * bi_direction]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # 把句子序列再调整成输入时的顺序
        output = output[desorted_indices]
        # output = [batch_size,seq_len,hidden_dim * num_directionns ]
        batch_size, max_seq_len, hidden_dim = output.shape
        # 拼接左右上下文信息
        output = torch.tanh(self.fc_cat(torch.cat((output, embedded), dim=2)))

        output = torch.transpose(output, 1, 2)
        output = F.max_pool1d(output, max_seq_len).squeeze().contiguous()
        output = self.fc(output)

        return output


class RnnAttentionModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers, bidirectional, dropout, word_embedding, freeze,
                 batch_first=True):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(self.hidden_dim * 2,self.hidden_dim*2))
        self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, _, text_lengths):
        # 按照句子长度从大到小排序
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text = [batch size,sent len]
        embedded = self.dropout(self.embedding(text)).to(torch.float32)
        # embedded = [batch size,sent len,  emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output = [sent len, batch size, hidden dim * num_direction]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # 把句子序列再调整成输入时的顺序
        output = output[desorted_indices]
        # attention
        # M = [sent len, batch size, hidden dim * num_direction]
        # M = self.tanh1(output)
        alpha = F.softmax(torch.matmul(self.tanh1(output), self.w), dim=0).unsqueeze(-1)  # dim=0表示针对文本中的每个词的输出softmax
        output_attention = output * alpha

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)

        output_attention = torch.sum(output_attention, dim=1)
        output = torch.sum(output, dim=1)

        fc_input = self.dropout(output + output_attention + hidden)
        # fc_input = self.dropout(output_attention)
        out = self.fc(fc_input)
        return out


class FastText(BaseModel):
    def __init__(self, output_dim, word_embedding, freeze):
        super().__init__()
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        self.fc = nn.Linear(self.embedding_size, output_dim)

    def forward(self, text, _, text_lengths):
        # text = [batch size,sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        # pooled = [batch size, embedding_dim]
        return self.fc(pooled)


class TextCNN(BaseModel):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, word_embedding, freeze):
        # n_filter 每个卷积核的个数
        super().__init__()
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, self.embedding_size)) for fs in
             filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
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


class TextCNN1d(BaseModel):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, word_embedding, freeze):
        super().__init__()
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embedding_size, out_channels=n_filters, kernel_size=fs) for fs in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
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


class HierAttNet(BaseModel):
    def __init__(self,rnn_type, word_hidden_size, sent_hidden_size, num_classes, word_embedding,n_layers,bidirectional,batch_first,freeze,dropout):
        super(HierAttNet, self).__init__()
        self.word_embedding = word_embedding
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttNet(rnn_type,word_embedding, word_hidden_size,n_layers,bidirectional,batch_first,dropout,freeze)
        self.sent_att_net = SentAttNet(rnn_type,sent_hidden_size, word_hidden_size,n_layers,bidirectional,batch_first,dropout, num_classes)


    def forward(self, batch_doc, _, text_lengths):
        output_list = []
        # ############################ 词级 #########################################
        for idx,doc in enumerate(batch_doc):
            # 把一篇文档拆成多个句子
            doc = doc[:text_lengths[idx]]
            doc_list = doc.cpu().numpy().tolist()
            sep_index = [i for i, num in enumerate(doc_list) if num == self.word_embedding.stoi['[SEP]']]
            sentence_list = []
            if sep_index:
                pre = 0
                for cur in sep_index:
                    sentence_list.append(doc_list[pre:cur])
                    pre = cur

                sentence_list.append(doc_list[cur:])

            else:
                sentence_list.append(doc_list)
            max_sentence_len = len(max(sentence_list,key=lambda x:len(x)))
            seq_lens = []
            input_token_ids = []
            for sent in sentence_list:
                cur_sent_len = len(sent)
                seq_lens.append(cur_sent_len)
                input_token_ids.append(sent+[self.word_embedding.stoi['PAD']]*(max_sentence_len-cur_sent_len))
            input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(batch_doc.device)
            seq_lens = torch.LongTensor(np.array(seq_lens)).to(batch_doc.device)
            word_output, hidden = self.word_att_net(input_token_ids,seq_lens)
            # word_output = [bs,hidden_size]
            output_list.append(word_output)

        max_doc_sent_num = len(max(output_list,key=lambda x: len(x)))
        batch_sent_lens = []
        batch_sent_inputs = []

        # ############################ 句子级 #########################################
        for doc in output_list:
            cur_doc_sent_len = len(doc)
            batch_sent_lens.append(cur_doc_sent_len)
            expand_doc = torch.cat([doc,torch.zeros(size=((max_doc_sent_num-cur_doc_sent_len),len(doc[0]))).to(doc.device)],dim=0)
            batch_sent_inputs.append(expand_doc.unsqueeze(dim=0))

        batch_sent_inputs = torch.cat(batch_sent_inputs, 0)
        batch_sent_lens = torch.LongTensor(np.array(batch_sent_lens)).to(doc.device)
        output = self.sent_att_net(batch_sent_inputs,batch_sent_lens)
        return output

class WordAttNet(BaseModel):
    def __init__(self, rnn_type, word_embedding, hidden_size, n_layers, bidirectional, batch_first, dropout,freeze=True):
        super(WordAttNet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.rnn_type = rnn_type
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_size,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, text, text_lengths):

        # 按照句子长度从大到小排序
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text = [batch size,sent len]
        embedded = self.dropout(self.embedding(text)).to(torch.float32)
        # embedded = [batch size,sent len,  emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output = [sent len, batch size, hidden dim * num_direction]
        seq_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # 把句子序列再调整成输入时的顺序
        seq_output = seq_output[desorted_indices]

        output = matrix_mul(seq_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight)
        output = F.softmax(output,dim=-1)
        output = element_wise_mul(seq_output, output)

        return output, hidden


class SentAttNet(BaseModel):

    def __init__(self, rnn_type, sent_hidden_size, word_hidden_size, n_layers, bidirectional, batch_first, dropout,
                 num_classes):
        super(SentAttNet, self).__init__()
        self.batch_first = batch_first
        self.rnn_type = rnn_type
        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(2*word_hidden_size,
                               hidden_size=sent_hidden_size,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(2*word_hidden_size,
                               hidden_size=sent_hidden_size,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(2*word_hidden_size,
                               hidden_size=sent_hidden_size,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        self._create_weights(mean=0.0, std=0.05)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, sentence_tensor, text_lengths):

        # 按照句子长度从大到小排序
        sentence_tensor, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(sentence_tensor, text_lengths)
        # text = [batch size,sent len,  emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(sentence_tensor, sorted_seq_lengths, batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output = [sent len, batch size, hidden dim * num_direction]
        seq_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # 把句子序列再调整成输入时的顺序
        seq_output = seq_output[desorted_indices]
        output = matrix_mul(seq_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output,dim=-1)
        output = element_wise_mul(seq_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output

# ####################################bert#################################################################

class Bert(BaseModel):

    def __init__(self, bert_path, num_classes, word_embedding, trained=True):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # 不对bert进行训练
        for param in self.bert.parameters():
            param.requires_grad = trained

        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_classes)

    def forward(self, context, bert_masks, seq_lens):
        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        sentence, cls = self.bert(context, attention_mask=bert_masks)
        # sentence = torch.sum(sentence,dim=1)
        out = self.fc(cls)
        return out


class BertCNN(nn.Module):

    def __init__(self, bert_path, num_filters, hidden_size, filter_sizes, dropout, num_classes, word_embedding,
                 trained=True):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = trained
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, hidden_size)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)

        self.fc_cnn = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, context, bert_masks, seq_len):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=bert_masks)
        encoder_out = self.dropout(encoder_out)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class BertRNN(nn.Module):

    def __init__(self, rnn_type, bert_path, hidden_dim, n_layers, bidirectional, batch_first, word_embedding,
                 dropout, num_classes, trained):
        super(BertRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = trained
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.bert.config.to_dict()['hidden_size'],
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text, bert_masks, seq_lens):

        # text = [batch size,sent len]
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        bert_sentence, bert_cls = self.bert(text, attention_mask=bert_masks)
        # 按照句子长度从大到小排序
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(bert_sentence, seq_lens)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths,
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # output = [ batch size,sent len, hidden_dim * bidirectional]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.sum(output, dim=1)
        fc_input = self.dropout(output + hidden)
        out = self.fc_rnn(fc_input)

        return out


class BertRCNN(BaseModel):
    def __init__(self, rnn_type, bert_path, hidden_dim, n_layers, bidirectional, dropout, num_classes, word_embedding,
                 trained, batch_first=True):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = trained

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.bert.config.to_dict()['hidden_size'],
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        # self.maxpool = nn.MaxPool1d()
        self.fc = nn.Linear(hidden_dim * n_layers, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, bert_masks, seq_lens):
        # text = [batch size,sent len]
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        bert_sentence, bert_cls = self.bert(text, attention_mask=bert_masks)
        sentence_len = bert_sentence.shape[1]
        bert_cls = bert_cls.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        bert_sentence = bert_sentence + bert_cls
        # 按照句子长度从大到小排序
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(bert_sentence, seq_lens)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths,
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]
        batch_size, max_seq_len, hidden_dim = output.shape
        out = torch.transpose(output.relu(), 1, 2)

        out = F.max_pool1d(out, max_seq_len).squeeze()
        out = self.fc(out)

        return out


# ####################################### xlnet ######################################################################

class XLNet(BaseModel):

    def __init__(self, xlnet_path, num_classes, word_embedding, trained=True):
        super(XLNet, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(xlnet_path)
        # 不对bert进行训练
        for param in self.xlnet.parameters():
            param.requires_grad = trained

        self.fc = nn.Linear(self.xlnet.d_model, num_classes)

    def forward(self, context, xlnet_masks, seq_lens):
        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        sentence_encoder = self.xlnet(context, attention_mask=xlnet_masks)
        sentence_encoder = torch.sum(sentence_encoder[0], dim=1)
        out = self.fc(sentence_encoder)
        return out
