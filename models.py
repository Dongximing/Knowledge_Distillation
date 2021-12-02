import torch
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import config
import torch.nn.functional as F
config.seed_torch()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRUBaseline(nn.Module):
    def __init__(self,vocab_size,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding_dim =100):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings =vocab_size,embedding_dim= embedding_dim,padding_idx=1)
        self.hidden_size = hidden_dim
        self.rnn = nn.GRU(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout, bidirectional=bidirectional,batch_first=True)
        self.fc = nn.Linear(hidden_dim*2,number_class)
        self.dropout = nn.Dropout(dropout)
    def forward(self,text,text_length):

        a_lengths, idx = text_length.sort(0, descending=True)
        _, un_idx = t.sort(idx, dim=0)
        seq = text[idx]
        # print(text)
        seq = self.embedding_layer(seq)
        # seq = self.embedding_layer(seq)

        # print(seq)

        a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=a_lengths.to('cpu'), batch_first=True)
        packed_output, (hidden) = self.rnn(a_packed_input)
        out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = self.dropout(t.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = t.index_select(out, 0, un_idx)
        hidden = t.index_select(hidden, 0, un_idx)

        output = self.fc(hidden)
class LSTMBaseline(nn.Module):
    def __init__(self,vocab_size,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding_dim =100):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings =vocab_size,embedding_dim= embedding_dim,padding_idx=1)
        self.hidden_size = hidden_dim
        self.rnn = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout, bidirectional=bidirectional,batch_first=True)
        self.fc = nn.Linear(hidden_dim*2,number_class)
        self.dropout = nn.Dropout(dropout)
    def forward(self,text,text_length):

        # a_lengths, idx = text_length.sort(0, descending=True)
        # _, un_idx = t.sort(idx, dim=0)
        # seq = text[idx]
        # print(text)
        seq = self.dropout(self.embedding_layer(text))
        # seq = self.embedding_layer(seq)

        # print(seq)

        # a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=text_length.to('cpu'), batch_first=True,enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(seq)
        # out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # hidden = self.dropout(packed_output[:,-1,:])
        out = torch.cat([packed_output[:, -1, :self.hidden_size], packed_output[:, 0, self.hidden_size:]], 1)



        output = self.fc(out)





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
        self.fc = nn.Linear(hidden_dim*2 , number_class)
        self.dropout = nn.Dropout(dropout)
        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.lstm_dropout = nn.Dropout(0.5)
        self.w = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size*4, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, number_class)
        )


    def attention_net_with_w(self, lstm_out, lstm_hidden):

        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
            # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
            # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
            # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
            # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
            # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
            # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
            # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
            # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result


        # self.w_omega = nn.Parameter(torch.Tensor(
        #     hidden_dim * 2, hidden_dim* 2))
        # self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        # self.decoder = nn.Linear(2 * hidden_dim, 2)
        #
        # nn.init.uniform_(self.w_omega, -0.1, 0.1)
        # nn.init.uniform_(self.u_omega, -0.1, 0.1)
    def atten(self,output,finial_state):
        # merged_state = torch.sum(finial_state,dim=0)
        # merged_state = finial_state[-1,:,:]+finial_state[-2,:,:]
        # merged_state = finial_state.squeeze(0)
        finial_state =finial_state.unsqueeze(2)
        attent_weight = torch.bmm(output,finial_state).squeeze(2)
        soft_max_weights = F.softmax(attent_weight,1)
        context = torch.bmm(output.transpose(1,2),soft_max_weights.unsqueeze(2)).squeeze(2)
        return context
    def attention(self,finial_state,mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)
        h = self.tanh(finial_state)  # (batch_size, word_pad_len, rnn_size)
        att_score = torch.bmm(self.tanh(h), att_weight)
        # eq.10: α = softmax(w^T M)
        mask = mask.unsqueeze(dim=-1)
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))
        att_weight = F.softmax(att_score, dim=1)

        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L *  B*L*1 -> B*H*1 -> B*H
        reps = self.tanh(reps)

        # alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        # alpha = self.softmax(alpha)  # (batch_size, word_pad_len)
        #
        # r = finial_state * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        # r = r.sum(dim = 1)  # (batch_size, rnn_size)

        return reps

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(2* 2, b_size, self.hidden_size))
        c0 = Variable(torch.zeros(2* 2, b_size, self.hidden_size))

        h0 = h0.to(device)
        c0 = c0.to(device)
        return (h0, c0)
    def forward(self,text,text_length,mask):

        # a_lengths, idx = text_length.sort(0, descending=True)
        # _, un_idx = t.sort(idx, dim=0)
        # seq = text[idx]
        self.hidden = self.init_hidden(text.size(0))

        seq = self.dropout(self.embedding_layer(text))
        a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=text_length.to('cpu'), batch_first=True,enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(a_packed_input,self.hidden)
        out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # batch_size = hidden.shape[1]
        # h_n_final_layer = hidden.view(2,
        #                            2,
        #                            batch_size,
        #                            256)[-1, :, :, :]
        # out = out.view(-1, self.max_len, 2, self.hidden_size)
        # out = torch.sum(out, dim=2)
        # u = torch.tanh(torch.matmul(out, self.w_omega))
        # # u形状是(batch_size, seq_len, 2 * num_hiddens)
        # att = torch.matmul(u, self.u_omega)
        # # att形状是(batch_size, seq_len, 1)
        # att_score = F.softmax(att, dim=1)
        # # att_score形状仍为(batch_size, seq_len, 1)
        # scored_x = out * att_score
        # # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # # Attention过程结束

        # context = torch.sum(scored_x, dim=1)


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
        hidden = self.dropout(t.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = self.dropout(t.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # # print(hidden.size())
        # out = out[:, :, : self.hidden_size] + out[:, :, self.hidden_size:]
        # context, alphas = self.attention(H)
        # context = self.tanh(context)

        # context = self.attention(out,mask)
        # hidden = hidden.permute(1, 0, 2)
        # final_hidden_state = torch.cat([h_n_final_layer[i, :, :] for i in range(h_n_final_layer.shape[0])], dim=1)
        # out =self.dropout(out)
        context = self.atten(out, hidden)
        concatenated_vector = torch.cat([hidden, context], dim=1)
        # concatenated_vector =self.dropout(concatenated_vector)

        # out = t.index_select(out, 0, un_idx)
        # context = t.index_select(context, 0, un_idx)
        # context = self.lstm_dropout(context)
        return self.fc_out(concatenated_vector)
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






