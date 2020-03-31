# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 4:47 下午
# @Author  : lizhen
# @FileName: model.py
# @Description:
import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from pytorch_transformers import BertModel
class DPCNN(nn.Module):
    def __init__(self, vocab, embedding_dim, num_filters, use_pretrain_embedding,num_classes):
        super(DPCNN, self).__init__()
        if use_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        else:
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.pad_index)
        self.conv_region = nn.Conv2d(1, num_filters, (3, embedding_dim), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, text,text_lengths):
        # text [batch_size,seq_len]
        x = self.embedding(text) # x=[batch_size,seq_len,embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        x = self.conv_region(x)  # x = [batch_size, num_filters, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x) # [batch_size, num_filters,1,1]
        x = x.squeeze()  # [batch_size, num_filters]
        x = self.fc(x) # [batch_size, 1]
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


class RCNNModel(BaseModel):
    def __init__(self, rnn_type, vocab, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_size=32,batch_first=False,use_pretrain_embedding=False):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.pad_size = pad_size
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

        # self.maxpool = nn.MaxPool1d()
        self.fc = nn.Linear(hidden_dim * n_layers + embedding_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=self.batch_first)

        # packed_output
        # hidden [n_layers * bi_direction,batch_size,hidden_dim]
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output [sent len, batch_size * n_layers * bi_direction]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # if not self.bidirectional:
        #     hidden = torch.reshape(hidden,(hidden.shape[1],self.hidden_dim * self.n_layers))
        # else:
        #     hidden = torch.reshape(hidden, (-1,hidden.shape[1], self.hidden_dim * self.n_layers))
        #     hidden = torch.mean(hidden,dim=0)

        output = torch.cat((output,embedded),2)
        out = output.relu().permute(1,2,0)
        max_sentence_len = output_lengths[0].item()
        out = nn.MaxPool1d(max_sentence_len)(out).squeeze()
        out = self.fc(out)

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
        # n_filter 每个卷积核的个数
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




class RnnAttentionModel(BaseModel):
    def __init__(self, rnn_type, vocab, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_first=False,use_pretrain_embedding=False):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.batch_first = batch_first

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


        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(self.hidden_dim * 2,self.hidden_dim*2))
        self.w = nn.Parameter(torch.randn(hidden_dim),requires_grad=True)

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

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
        # output = [sent len, batch size, hidden dim * num_direction]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # attention
        # M = [sent len, batch size, hidden dim * num_direction]
        # M = self.tanh1(output)

        alpha = F.softmax(torch.matmul(self.tanh1(output), self.w), dim=0).unsqueeze(-1) # dim=0表示针对文本中的每个词的输出softmax
        output_attention = output * alpha

        # hidden = [n_layers * num_direction,batch_size, hidden_dim]
        if self.bidirectional:
            hidden = torch.mean(torch.reshape(hidden, (-1,hidden.shape[1], self.hidden_dim * 2)),dim=0)  # hidden = [batch_size, hidden_dim * num_direction]
        else:
            hidden = torch.mean(torch.reshape(hidden, (-1, hidden.shape[1], self.hidden_dim)), dim=0)   # hidden = [batch_size, hidden_dim]

        output_attention = torch.sum(output_attention,dim=0)
        output = torch.sum(output,dim=0)



        fc_input = self.dropout(output+output_attention+hidden)
        # fc_input = self.dropout(output_attention)
        out = self.fc(fc_input)

        return out

class Bert(nn.Module):

    def __init__(self, bert_model_path,num_classes):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        # 不对bert进行训练
        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'],num_classes)

    def forward(self, context, seq_len, mask):

        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        sentence, cls = self.bert(context,attention_mask=mask)

        out = self.fc(cls)
        return out

class BertCNN(nn.Module):

    def __init__(self, bert_model_path,num_filters,hidden_size,filter_sizes,dropout,num_classes):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, hidden_size)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)

        self.fc_cnn = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, context, seq_len, mask):
        #context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out

class BertRNN(nn.Module):

    def __init__(self,rnn_type,bert_model_path,bert_embedding_dim,hidden_dim ,n_layers,bidirectional,batch_first,dropout,num_classes):
        super(BertRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bert = BertModel.from_pretrained(bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(bert_embedding_dim,
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(bert_embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(bert_embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)


        self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, context, seq_len, mask):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask)
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(encoder_out)
        else:
            output, (hidden, cell) = self.rnn(encoder_out)


        if not self.bidirectional:
            hidden = torch.reshape(hidden,(hidden.shape[1],self.hidden_dim * self.n_layers))
        else:
            hidden = torch.reshape(hidden, (-1,hidden.shape[1], self.hidden_dim * self.n_layers))
            hidden = torch.mean(hidden,dim=0)
        output = torch.sum(output,dim=1)
        fc_input = self.dropout(output+hidden)
        out = self.fc_rnn(fc_input)

        return out


class BertRCNN(BaseModel):
    def __init__(self, rnn_type, bert_model_path, bert_embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout,num_classes, batch_first=False):
        super().__init__()
        self.rnn_type = rnn_type.lower()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bert_embedding_dim = bert_embedding_dim
        self.bert = BertModel.from_pretrained(bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(bert_embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(bert_embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(bert_embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        # self.maxpool = nn.MaxPool1d()
        self.fc = nn.Linear(hidden_dim * n_layers + bert_embedding_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, context, seq_len, mask):
        # text = [sent len, batch size]

        encoder_out, text_cls = self.bert(context, attention_mask=mask)
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(encoder_out)
        else:
            output, (hidden, cell) = self.rnn(encoder_out)


        output = torch.cat((output,encoder_out),2)
        out = output.relu().permute(0,2,1)
        max_sentence_len = torch.max(seq_len).item()
        out = nn.MaxPool1d(max_sentence_len)(out).squeeze()
        out = self.fc(out)

        return out
