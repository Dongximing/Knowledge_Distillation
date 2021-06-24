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
        # seq = self.dropout(self.embedding(text))
        # _, hidden = self.rnn(seq)
        #
        # # hidden = [n layers * n directions, batch size, emb dim]
        #
        # if self.rnn.bidirectional:
        #     hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # else:
        #     hidden = self.dropout(hidden[-1, :, :])
        a_lengths, idx = text_length.sort(0, descending=True)
        _, un_idx = t.sort(idx, dim=0)
        seq = text[idx]

        seq = self.dropout(self.embedding(seq))

        a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=a_lengths.to('cpu'), batch_first=True)
        packed_output, (hidden, cell) = self.rnn(a_packed_input)
        out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = self.dropout(t.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        hidden = t.index_select(hidden, 0, un_idx)

        # hidden = [batch size, hid dim]

        output = self.fc(hidden)

        # text = text.permute(1,0)
        # seq_size, batch_size = text.size(0), text.size(1)
        # length_perm = (-text_length).argsort()
        # length_perm_inv = length_perm.argsort()
        # seq = torch.gather(text, 1, length_perm[None, :].expand(seq_size, batch_size))
        # length = torch.gather(text_length, 0, length_perm)
        # # Pack sequence
        # seq = self.dropout(self.embedding(seq))
        # seq = nn.utils.rnn.pack_padded_sequence(seq, length.to('cpu'))
        # # Send through LSTM
        # features, hidden_states = self.rnn(seq)
        # # Unpack sequence
        # features = nn.utils.rnn.pad_packed_sequence(features)[0]
        # # Separate last dimension into forward/backward features
        # features = features.view(seq_size, batch_size, 2, -1)
        # # Index to get forward and backward features and concatenate
        # # Gather last word for each sequence
        # last_indexes = (length - 1)[None, :, None, None].expand((1, batch_size, 2, features.size(-1)))
        # last_indexes = torch.tensor(last_indexes,dtype=torch.int64)
        # forward_features = torch.gather(features, 0, last_indexes)
        # # Squeeze seq dimension, take forward features
        # forward_features = forward_features[0, :, 0]
        # # Take first word, backward features
        # backward_features = features[0, :, 1]
        # features = torch.cat((forward_features, backward_features), -1)
        #
        # logits = self.fc(features)
        # logits = torch.gather(logits, 0, length_perm_inv[:, None].expand((batch_size, logits.size(-1))))

        # _, idx = text_length.sort(0, descending=True)
        # text = text[idx]
        # embedded = self.dropout(self.embedding(text))
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,lengths=text_length.to('cpu'),batch_first = True)
        # output_packed,(hidden,cell) = self.rnn(packed_embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(output_packed,batch_first = True)
        # idx = (torch.LongTensor(text_length) - 1).view(-1, 1).expand(len(text_length), output.size(2))
        # time_dimension = 1 if text_length else 0
        # idx = idx.unsqueeze(time_dimension)
        # if output.is_cuda:
        #     idx = idx.cuda(output.data.get_device())
        # # Shape: (batch_size, rnn_hidden_dim)
        # last_output = output.gather(time_dimension, Variable(idx)).squeeze(time_dimension)


        return output






