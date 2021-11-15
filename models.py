import torch
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import config
import torch.nn.functional as F
config.seed_torch()
class LSTMBaseline(nn.Module):
    def __init__(self,vocab_size,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding_dim =100):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings =vocab_size,embedding_dim= embedding_dim,padding_idx=1)
        self.hidden_size = hidden_dim
        self.rnn = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout, bidirectional=bidirectional,batch_first=True)
        # self.rnn = nn.GRU(embedding_dim,
        #                   hidden_dim,
        #                   num_layers=n_layers,
        #                   bidirectional=bidirectional,
        #                   batch_first=True,
        #                   dropout=0 if n_layers < 2 else dropout)
        self.fc = nn.Linear(hidden_dim*2,number_class)
        self.dropout = nn.Dropout(dropout)
    def forward(self,text,text_length):

        a_lengths, idx = text_length.sort(0, descending=True)
        _, un_idx = t.sort(idx, dim=0)
        seq = text[idx]
        # print(text)
        seq = self.dropout(self.embedding_layer(seq))
        # seq = self.embedding_layer(seq)

        # print(seq)

        a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=a_lengths.to('cpu'), batch_first=True)
        packed_output, (hidden, cell) = self.rnn(a_packed_input)
        out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = self.dropout(t.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = t.index_select(out, 0, un_idx)
        hidden = t.index_select(hidden, 0, un_idx)
        # last_timesteps =text_length.unsqueeze(1)
        # last_timesteps = last_timesteps-1
        # last_timesteps =torch.tensor(last_timesteps, dtype=torch.int64).to(device = 'cuda')
        #
        # relative_hidden_size = self.hidden_size * 2
        # last_timesteps = last_timesteps.repeat(1, relative_hidden_size)  # (1, B x H*)
        # last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)  # (B, 1, H*)
        #
        # pooled_sequence_output = out.gather(  # (B, H*)
        #     dim=1,
        #     index=last_timesteps
        # ).squeeze()


        # hidden = [batch size, hid dim]

        output = self.fc(hidden)





        return output
class CNN_Baseline(nn.Module):
    def __init__(self, vocab_size,nKernel,ksz,number_class,embedding_dim =100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=1)
        self.convs = nn.ModuleList([nn.Conv2d(1,nKernel,(k,embedding_dim)) for k in ksz])
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(len(self.ksz)*self.nKernel,number_class)
    def forward(self,text):
        embedding = self.embedding_layer(text)
        embedding = embedding.unsqueeze(1)
        x_convs = [F.relu(conv(embedding)).squeeze(3) for conv in self.convs]
        x_maxpool = [F.max_pool1d(x_conv,x_conv.size(2)).squeeze(2) for x_conv in x_convs ]
        flatten = torch.cat(x_maxpool, 1)
        x = self.dropout(flatten)
        x = self.linear(x)
        return x



class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']
        self.LSTM = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout, bidirectional=bidirectional,batch_first=True)

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask):
        with torch.no_grad():

            embedded = self.dropout(self.bert(ids, attention_mask=mask)[0])

        # embedded = [batch size, sent len, emb dim]
        # output,(hidden,ct) = self.rnn(embedded)
        _, hidden = self.rnn(embedded)
        # print(hidden.shape)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)


        return output

class LSTM_atten(nn.Module):
    def __init__(self,vocab_size,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding_dim =100):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=1)
        self.hidden_size = hidden_dim
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim , number_class)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
    def atten(self,output,finial_state):
        attent_weight = torch.bmm(output,finial_state).squeeze(2)
        soft_max_weights = F.softmax(attent_weight,1)
        context = torch.bmm(output.transpose(1,2),soft_max_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self,text,text_length):

        a_lengths, idx = text_length.sort(0, descending=True)
        _, un_idx = t.sort(idx, dim=0)
        seq = text[idx]

        seq = self.dropout(self.embedding_layer(seq))
        a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=a_lengths.to('cpu'), batch_first=True)
        packed_output, (hidden, cell) = self.rnn(a_packed_input)
        out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # (forward_out, backward_out) = torch.chunk(out, 2, dim=2)
        # out = forward_out + backward_out  # [seq_len, batch, hidden_size]



        # 为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重  # [batch, num_layers * num_directions,  hidden_size]
        # (hidden_f, hidden) = torch.chunk(hidden, 2, dim=0)
        # hidden = hidden.permute(1, 0, 2)
        # hidden = t.sum(hidden, dim=1)
        #    # [batch, 1,  hidden_size]
        # h_n = hidden.squeeze(dim=1)  # [batch, hidden_size]
        #
        # attention_w = self.attention_weights_layer(h_n)  # [batch, hidden_size]
        # attention_w = attention_w.unsqueeze(dim=1)  # [batch, 1, hidden_size]
        #
        # attention_context = torch.bmm(attention_w, out.transpose(1, 2))  # [batch, 1, seq_len]
        # softmax_w = F.softmax(attention_context, dim=-1)  # [batch, 1, seq_len],权重归一化
        #
        # x = torch.bmm(softmax_w, out)  # [batch, 1, hidden_size]
        # x = x.squeeze(dim=1)  # [batch, hidden_size]
        # x = self.fc(x)
        # return x
        hidden = self.dropout(t.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)).unsqueeze(2)
        # print(hidden.size())
        context = self.atten(out,hidden)

        out = t.index_select(out, 0, un_idx)
        context = t.index_select(context, 0, un_idx)
        context = self.dropout(context)
        return self.fc(context)
class BERT(nn.Module):
    def __init__(self,bert):

        super(BERT, self).__init__()

        self.bert = bert
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[1]
        # bo = self.bert_drop(o2)
        output = self.out(o2)
        return output


# import torch
# import torch.nn as nn
#
# class BERTGRUSentiment(nn.Module):
#     def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,dropout):
#
#         super().__init__()
#
#         self.bert = bert
#
#         embedding_dim = bert.config.to_dict()['hidden_size']
#
#         self.rnn = nn.LSTM(embedding_dim,
#                           hidden_dim,
#                           num_layers=n_layers,
#                           bidirectional=bidirectional,
#                           batch_first=True,
#                           dropout=0 if n_layers < 2 else dropout)
#
#         self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text,):
#
#         # text = [batch size, sent len]
#
#         with torch.no_grad():
#             embedded = self.bert(text)[0]
#
#         # embedded = [batch size, sent len, emb dim]
#
#         output, (hidden,h_c) = self.rnn(embedded)
#
#         # hidden = [n layers * n directions, batch size, emb dim]
#         # print(hidden.size())
#         if self.rnn.bidirectional:
#             hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         else:
#             hidden = self.dropout(hidden[-1, :, :])
#
#         # hidden = [batch size, hid dim]
#
#         output = self.out(hidden)
#
#         # output = [batch size, out dim]
#
#         return output
# class Dilated_CNN(nn.Module):
#     def __init__(self,):






