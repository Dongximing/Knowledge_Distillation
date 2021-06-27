
# print(t[0,-5:])
import torch
from torch import LongTensor
from torch.nn import Embedding, LSTM
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
config.seed_torch()



# We want to run LSTM on a batch following 3 character sequences
seqs = ['long_str',  # len = 8
        'tiny',      # len = 4
        'medium']    # len = 6


## Step 1: Construct Vocabulary ##
##------------------------------##
# make sure <pad> idx is 0
vocab = ['<pad>'] + sorted(set([char for seq in seqs for char in seq]))



## Step 2: Load indexed data (list of instances, where each instance is list of character indices) ##
##-------------------------------------------------------------------------------------------------##
vectorized_seqs = [[vocab.index(tok) for tok in seq]for seq in seqs]

embed = Embedding(len(vocab), 4) # embedding_dim = 4
lstm = LSTM(input_size=4, hidden_size=5, batch_first=True,bidirectional=True,num_layers=2) # input_dim = 4, hidden_dim = 5




# get the length of each seq in your batch
seq_lengths = LongTensor(list(map(len, vectorized_seqs)))


seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
# seq_tensor => [[0 0 0 0 0 0 0 0]
#                [0 0 0 0 0 0 0 0]
#                [0 0 0 0 0 0 0 0]]

for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = LongTensor(seq)


seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
_, un_idx = torch.sort(perm_idx, dim=0)
seq_tensor = seq_tensor[perm_idx]


embedded_seq_tensor = embed(seq_tensor)


packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)


packed_output, (ht, ct) = lstm(packed_input)

output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
# print(ht)
output = torch.index_select(output, 0, un_idx)
# print(output[:,-1,:])
a = [7,3,5]

print(a)
a= torch.tensor(a).unsqueeze(1)
print(a-1)
a = a.repeat(1,10)
# print(a
a = a.view(-1, 1,10)
# print(a.shape)
# print(a)
# print(output)
pooled_sequence_output = output.gather(                      # (B, H*)
            dim=1,
            index=a
        ).squeeze()
# print(pooled_sequence_output)
# hidden = torch.index_select(ht, 0, un_idx)



# final = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1)
# print(final)
# final= torch.index_select(final, 0, un_idx)
# print(final)
# print(final.shape)
print(output)
print(pooled_sequence_output)
#


