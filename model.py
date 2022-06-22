import torch
import torch.nn as nn
import torch as t
import torch.nn.functional as F
from utils import IMDB_indexing, pad_sequencing
class LSTM_atten(nn.Module):
    def __init__(self,vocab_size,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding_dim =100):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=1)
        self.hidden_size = hidden_dim
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim*2 , number_class)
        self.dropout = nn.Dropout(dropout)
        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size*2, 1))

        self.tanh = nn.Tanh()




    def attention(self,finial_state,mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)
        h = self.tanh(finial_state)  
        att_score = torch.bmm(h, att_weight)
        mask = mask.unsqueeze(dim=-1)
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))
        att_weight = F.softmax(att_score, dim=1)


        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)
        reps = self.tanh(reps)



        return reps
    def forward(self,text,text_length,mask):



        seq = self.dropout(self.embedding_layer(text))
        a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=text_length.to('cpu'), batch_first=True,enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(a_packed_input)
        out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        context = self.attention(out,mask)


        return self.fc(context)

