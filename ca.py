#
# # print(t[0,-5:])
#
# import torch
# from torch import LongTensor
# from torch.nn import Embedding, LSTM
# from torch.autograd import Variable
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from nltk.corpus import stopwords
# from tqdm import tqdm
# from torch import nn as nn
#
# import config
# key_embedding = torch.zeros(0)
# x = torch.arange(3).reshape(1, 3)
# key_embedding = torch.cat((key_embedding, x), dim=0)
# key_embedding = torch.cat((key_embedding, x), dim=0)
# key_embedding = torch.mean(input=key_embedding, dim=0, keepdim=True)
# print(key_embedding)
#
#
# with tqdm(total=100) as pbar:
#     for i in range(10):
#         pbar.update(10)
# from torchtext.data.utils import get_tokenizer
# from torchtext.data.utils import ngrams_iterator
# from torchtext.vocab import Vocab
# from torchtext.vocab import build_vocab_from_iterator
# train_text = ["I fuck you", "I love you"]
# label =["D1","D2"]
#
# def _text_iterator(text, labels=None, ngrams=1, yield_label=False):
#     tokenizer = get_tokenizer('basic_english')
#     for i, text in enumerate(text):
#         texts = tokenizer(text)
#         filtered_text = [word for word in texts ]
#         if yield_label:
#             label = labels[i]
#             yield label, ngrams_iterator(filtered_text, ngrams)
#         else:
#             yield ngrams_iterator(filtered_text, ngrams)
# vocab = build_vocab_from_iterator(_text_iterator(train_text, label,1))
# print(vocab.get_stoi())
# def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):
#     # type: (List[Tensor], bool, float) -> Tensor
#     r"""Pad a list of variable length Tensors with ``padding_value``
#
#     ``pad_sequence`` stacks a list of Tensors along a new dimension,
#     and pads them to equal length. For example, if the input is list of
#     sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
#     otherwise.
#
#     `B` is batch size. It is equal to the number of elements in ``sequences``.
#     `T` is length of the longest sequence.
#     `L` is length of the sequence.
#     `*` is any number of trailing dimensions, including none.
#
#     Example:
#         >>> from torch.nn.utils.rnn import pad_sequence
#         >>> a = torch.ones(25, 300)
#         >>> b = torch.ones(22, 300)
#         >>> c = torch.ones(15, 300)
#         >>> pad_sequence([a, b, c]).size()
#         torch.Size([25, 3, 300])
#
#     Note:
#         This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
#         where `T` is the length of the longest sequence. This function assumes
#         trailing dimensions and type of all the Tensors in sequences are same.
#
#     Arguments:
#         sequences (list[Tensor]): list of variable length sequences.
#         batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
#             ``T x B x *`` otherwise
#         padding_value (float, optional): value for padded elements. Default: 0.
#
#     Returns:
#         Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
#         Tensor of size ``B x T x *`` otherwise
#     """
#
#     # assuming trailing dimensions and type of all the Tensors
#     # in sequences are same and fetching those from sequences[0]
#     max_size = sequences[0].size()
#     print("max_size:",max_size)
#     trailing_dims = max_size[1:]
#     max_len = max([s.size(0) for s in sequences])
#     if max_len < ksz:
#         max_len = ksz
#     if batch_first:
#         out_dims = (len(sequences), max_len) + trailing_dims
#     else:
#         out_dims = (max_len, len(sequences)) + trailing_dims
#
#     out_tensor = sequences[0].new_full(out_dims, padding_value)
#     for i, tensor in enumerate(sequences):
#         length = tensor.size(0)
#         # use index notation to prevent duplicate references to the tensor
#         if batch_first:
#             out_tensor[i, :length, ...] = tensor
#         else:
#             out_tensor[:length, i, ...] = tensor
#
#     return out_tensor
# list = []
# list1 = []
# i = torch.tensor([0.1,0.22,0.3])
# b = torch.tensor([0.3,0.4])
# d = torch.tensor([0.0,1,0.0])
# e = torch.tensor([1.0,0.0])
#
# d = torch.tensor([3,4,6,7,8,9,10,22,33,44,556,777])
# list.append(i)
#
# list.append(b)
# list1.append(d)
#
# list1.append(e)
# criterion = nn.CrossEntropyLoss()
# def compute_loss( scores, targets):
#     scores = scores.view(-1, scores.size(2))
#     loss = self.criterion(scores, targets.contiguous().view(-1))
#     return loss
# print(compute_loss(list, list1))
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
yt=mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
print(yt)
labels = []
for item in yt:
     label= np.where(item == 1)
     label =torch.tensor(label[0].tolist())
     print(label)
     labels.append(label)
print(labels)


print(labels)
def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):


    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if max_len < ksz:
        max_len = ksz
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
text = pad_sequence(labels, ksz=3, batch_first=True)
print(text)








# print(text)


# def fab(max):
#     n, a, b = 0, 0, 1
#     while n < max:
#         # yield b  # 使用 yield
#         print('b =',b)
#         a, b = b, a + b
#         n = n + 1
#
#
# for n in fab(5):
#     print (n)

# config.seed_torch()
# x = torch.arange(3)
# y = torch.arange(3)
# y=y.unsqueeze(1)
# x =x.unsqueeze(1)
# print(x.size())
#
# a = [ ]
# a.append(x)
# a.append(y)
# print(a)
# a = torch.stack(a,dim=1)
# print(a.size())
# print(a.squeeze(-1))
#
#
# a =torch.tensor([[0.1, 1.2,2.2, 3.1], [4.9, 5.2,0.0,0.1]])
# print(a.shape)
# a = torch.stack(a)
# inputs = a.split(1)[0].squeeze(0)
# print(a)
# print(inputs.size())
# dec = a[:,:-1]
# target = a[:,1:]
# print(dec)
# print("/n")
# print(target)

# for input in a.split(1,dim=1):
#   print(input.squeeze(0).size())


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
lstm = LSTM(input_size=4, hidden_size=5, batch_first=True,bidirectional=False,num_layers=2) # input_dim = 4, hidden_dim = 5




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


packed_output, state = lstm(packed_input)

output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)


# print(ht)
# (sat,cell) = state
# print(sat)
# print("/n")
# print(cell)
# print("/n")
# print(state)
# state = (state[0][::2], state[1][::2])
# print("/n")
# print(state)
# print("/n")
# output = torch.index_select(output, 0, un_idx)
# # print(output[:,-1,:])
# a = [7,3,5]
#
# print(a)
# a= torch.tensor(a).unsqueeze(1)
# print(a-1)
# a = a.repeat(1,10)
# # print(a
# a = a.view(-1, 1,10)
# # print(a.shape)
# # print(a)
# # print(output)
# pooled_sequence_output = output.gather(                      # (B, H*)
#             dim=1,
#             index=a
#         ).squeeze()
# print(pooled_sequence_output)
# hidden = torch.index_select(ht, 0, un_idx)



# final = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1)
# print(final)
# final= torch.index_select(final, 0, un_idx)
# print(final)
# print(final.shape)
# print(output)
# print(pooled_sequence_output)

# import torch
# import warnings
# warnings.filterwarnings('ignore')

# import torch.nn as nn
# import pandas as pd
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import train
# import config
# import dataloader
# from lstm import LSTMBaseline
# from earlystopping import EarlyStopping
# import os
# import ray
# config.seed_torch()
# from functools import partial
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
# def objective(x, a, b):
#     return a * (x ** 0.5) + b
# def trainable(config):
#     # config (dict): A dict of hyperparameters.
#
#     for x in range(20):
#         intermediate_score = objective(x, config["a"], config["b"])
#
#         tune.report(score=intermediate_score)  # This sends the score to Tune.
#
# analysis = tune.run(
#     trainable,
#     config={"a": 2, "b": 4}
# )
#
# print("best config: ", analysis.get_best_config(metric="score", mode="max"))

