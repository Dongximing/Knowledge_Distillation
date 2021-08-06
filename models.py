import torch
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import config
import torch.nn.functional as F
config.seed_torch()
class LSTMBaseline(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding,padding_idx=0)
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

        seq = self.dropout(self.embedding(seq))

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
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
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
        x = self.fc(x)
        return x
        
# class Dilated_CNN(nn.Module):
#     def __init__(self,):






