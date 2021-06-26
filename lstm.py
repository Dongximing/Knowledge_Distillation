import torch
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import config

config.seed_torch()
class LSTMBaseline(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding,padding_idx=0)
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
        # hidden = self.dropout(t.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        last_timesteps =text_length.unsqueeze(1)
        last_timesteps = last_timesteps-1
        relative_hidden_size = self.hidden_size * 2
        last_timesteps = last_timesteps.repeat(1, relative_hidden_size)  # (1, B x H*)
        last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)  # (B, 1, H*)
        pooled_sequence_output = out.gather(  # (B, H*)
            dim=1,
            index=last_timesteps
        ).squeeze()
        hidden = t.index_select(pooled_sequence_output, 0, un_idx)

        # hidden = [batch size, hid dim]

        output = self.fc(hidden)





        return output






