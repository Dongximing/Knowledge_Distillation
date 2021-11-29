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
import torch
import torch.nn as nn
# x = torch.rand(2,4,2)
# print(x)
# x = x.permute(0,2,1)
# print(x)
# conv = nn.Conv1d(in_channels=2,out_channels=2,kernel_size=2,padding=1,dilation=1)
# conv1 = nn.Conv1d(in_channels=2,out_channels=2,kernel_size=2,padding=1,dilation=2)
# conv2 = nn.Conv1d(in_channels=2,out_channels=2,kernel_size=2,padding=1,dilation=3)
# # print(conv.weight)
# # print((conv.weight).shape)
# # print(conv(x))
# # print((conv(x)).shape)
# print(x.shape)
#
# x = conv(x)
# print(x.shape)
# x = conv1(x)
# print(x.shape)
#
# x = conv2(x)
# print(x.shape)
# import csv
# train_data_path = '/Users/ximing/Desktop/IMDB-Movie-Reviews-50k-main/LSTM-baseline/valid.csv'
# text = []
# fields = []
# with open(train_data_path, 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     fields = next(csvreader)
#     for row in csvreader:
#         text.append(row[2])
# print(len(text))
import pandas as pd
# df = pd.read_csv(train_data_path)
# # saved_column = df.Review
# # for line in saved_column:
# #     text.append(line)
# # print(text)
# texts = []
# labels = []
# df = pd.read_csv(train_data_path)
# review = df.Review
# sentiment = df.Sentiment
# for text, label in zip(review, sentiment):
#     texts.append(text)
#     labels.append(label)
# print(texts)
# print(labels)
#
# from torchtext.data.utils import get_tokenizer
# from torchtext.data.utils import ngrams_iterator
# from torchtext.vocab import Vocab
# from torchtext.vocab import build_vocab_from_iterator
# train_text = ["I fuck you !", "I love you"]
# label =["D1","D2"]
#
# def _text_iterator(text, labels=None, ngrams=1, yield_label=False):
#     tokenizer = get_tokenizer('basic_english')
#     for i, text in enumerate(text):
#         texts = tokenizer(text)
#
#         filtered_text = [word for word in texts ]
#
#
#         yield ngrams_iterator(filtered_text, ngrams)
# vocab = build_vocab_from_iterator(_text_iterator(train_text, label,1))
#
# print(vocab.get_stoi())
# embedding = nn.Embedding(10, 3)
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# print(embedding.weight)
# # embedding1 = nn.Embedding(10, 3)
# # embedding1.weight.data.copy_(embedding.weight)
# embedding1 = nn.Embedding.from_pretrained(embedding.weight,freeze=False)
# print(embedding1.weight)

def pad_sequence(sequences, ksz, batch_first=False, padding_value=1):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``
    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.
    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.
    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])
    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.
    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()

    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    print(max_len)
    if max_len > ksz:
        max_len = ksz
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims



    out_tensor = sequences[0].new_full(out_dims, padding_value)
    mask_tensor = sequences[0].new_full(out_dims, 0)

    true =[]
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        print(length)
        if length > max_len:
            length = max_len
            out_tensor[i, :length, ...] = tensor[:length]
            mask_tensor[i, :length, ...] = torch.ones(length)
            true.append(length)
        else:
            out_tensor[i, :length, ...] = tensor[:length]
            mask_tensor[i, :length, ...] = torch.ones(length)
            true.append(length)


    return out_tensor, true,mask_tensor
import numpy as np
list = []
a = np.array([1,6,7])
b =np.array([1,2,3,1,1,1,1,1])
a = torch.from_numpy(a)

b = torch.from_numpy(b)
list.append(b)
list.append(a)


c, d,mask = pad_sequence(list, 5, batch_first=True, padding_value=10)
print(c)
print(d)
print(mask)
def a(a =10):
    c = a+10
    print(c)
a(20)
import torch
a = torch.rand(4,3,6)
print(a)
a = a.view(-1, 3, 2, 3)
a = torch.sum(a, dim=2)
print(a)
merged_state = torch.cat([s for s in a],1)
print(merged_state)
#
# from collections import Counter
# counter1 =  Counter({'x': 5, 'y': 12, 'z': -2, 'x1':0})
# counter2 = Counter({'x': 2, 'k':5})
# counter1.update(counter2)
# print(counter1)
# ls = [1,2,3]
# print(len(ls))
# lr = 1e-03
# print(lr)
#
# # def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):
# #     # type: (List[Tensor], bool, float) -> Tensor
# #     r"""Pad a list of variable length Tensors with ``padding_value``
# #
# #     ``pad_sequence`` stacks a list of Tensors along a new dimension,
# #     and pads them to equal length. For example, if the input is list of
# #     sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
# #     otherwise.
# #
# #     `B` is batch size. It is equal to the number of elements in ``sequences``.
# #     `T` is length of the longest sequence.
# #     `L` is length of the sequence.
# #     `*` is any number of trailing dimensions, including none.
# #
# #     Example:
# #         >>> from torch.nn.utils.rnn import pad_sequence
# #         >>> a = torch.ones(25, 300)
# #         >>> b = torch.ones(22, 300)
# #         >>> c = torch.ones(15, 300)
# #         >>> pad_sequence([a, b, c]).size()
# #         torch.Size([25, 3, 300])
# #
# #     Note:
# #         This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
# #         where `T` is the length of the longest sequence. This function assumes
# #         trailing dimensions and type of all the Tensors in sequences are same.
# #
# #     Arguments:
# #         sequences (list[Tensor]): list of variable length sequences.
# #         batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
# #             ``T x B x *`` otherwise
# #         padding_value (float, optional): value for padded elements. Default: 0.
# #
# #     Returns:
# #         Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
# #         Tensor of size ``B x T x *`` otherwise
# #     """
# #
# #     # assuming trailing dimensions and type of all the Tensors
# #     # in sequences are same and fetching those from sequences[0]
# #     max_size = sequences[0].size()
# #     print("max_size:",max_size)
# #     trailing_dims = max_size[1:]
# #     max_len = max([s.size(0) for s in sequences])
# #     if max_len < ksz:
# #         max_len = ksz
# #     if batch_first:
# #         out_dims = (len(sequences), max_len) + trailing_dims
# #     else:
# #         out_dims = (max_len, len(sequences)) + trailing_dims
# #
# #     out_tensor = sequences[0].new_full(out_dims, padding_value)
# #     for i, tensor in enumerate(sequences):
# #         length = tensor.size(0)
# #         # use index notation to prevent duplicate references to the tensor
# #         if batch_first:
# #             out_tensor[i, :length, ...] = tensor
# #         else:
# #             out_tensor[:length, i, ...] = tensor
# #
# #     return out_tensor
# # list = []
# # list1 = []
# # i = torch.tensor([0.1,0.22,0.3])
# # b = torch.tensor([0.3,0.4])
# # d = torch.tensor([0.0,1,0.0])
# # e = torch.tensor([1.0,0.0])
# #
# # d = torch.tensor([3,4,6,7,8,9,10,22,33,44,556,777])
# # list.append(i)
# #
# # list.append(b)
# # list1.append(d)
# #
# # list1.append(e)
# # criterion = nn.CrossEntropyLoss()
# # def compute_loss( scores, targets):
# #     scores = scores.view(-1, scores.size(2))
# #     loss = self.criterion(scores, targets.contiguous().view(-1))
# #     return loss
# # print(compute_loss(list, list1))
# # import numpy as np
# # import torch
# #
# # from sklearn.preprocessing import MultiLabelBinarizer
# #
# # print(yt)
# # labels = []
# #
# #
# #
# # print(labels)
# # def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):
# #
# #
# #     max_size = sequences[0].size()
# #     trailing_dims = max_size[1:]
# #     max_len = max([s.size(0) for s in sequences])
# #     if max_len < ksz:
# #         max_len = ksz
# #     if batch_first:
# #         out_dims = (len(sequences), max_len) + trailing_dims
# #     else:
# #         out_dims = (max_len, len(sequences)) + trailing_dims
# #
# #     out_tensor = sequences[0].new_full(out_dims, padding_value)
# #     for i, tensor in enumerate(sequences):
# #         length = tensor.size(0)
# #         # use index notation to prevent duplicate references to the tensor
# #         if batch_first:
# #             out_tensor[i, :length, ...] = tensor
# #         else:
# #             out_tensor[:length, i, ...] = tensor
# #
# #     return out_tensor
# # text = pad_sequence(labels, ksz=3, batch_first=True)
# # print(text)
#
#
#
#
#
#
#
#
# # print(text)
#
#
# # def fab(max):
# #     n, a, b = 0, 0, 1
# #     while n < max:
# #         # yield b  # 使用 yield
# #         print('b =',b)
# #         a, b = b, a + b
# #         n = n + 1
# #
# #
# # for n in fab(5):
# #     print (n)
#
# # config.seed_torch()
# # x = torch.arange(3)
# # y = torch.arange(3)
# # y=y.unsqueeze(1)
# # x =x.unsqueeze(1)
# # print(x.size())
# #
# # a = [ ]
# # a.append(x)
# # a.append(y)
# # print(a)
# # a = torch.stack(a,dim=1)
# # print(a.size())
# # print(a.squeeze(-1))
# #
# #
# # a =torch.tensor([[0.1, 1.2,2.2, 3.1], [4.9, 5.2,0.0,0.1]])
# # print(a.shape)
# # a = torch.stack(a)
# # inputs = a.split(1)[0].squeeze(0)
# # print(a)
# # print(inputs.size())
# # dec = a[:,:-1]
# # target = a[:,1:]
# # print(dec)
# # print("/n")
# # print(target)
#
# # for input in a.split(1,dim=1):
# #   print(input.squeeze(0).size())
#
#
# # We want to run LSTM on a batch following 3 character sequences
# seqs = ['long_str',  # len = 8
#         'tiny',      # len = 4
#         'medium']    # len = 6
#
#
# ## Step 1: Construct Vocabulary ##
# ##------------------------------##
# # make sure <pad> idx is 0
# vocab = ['<pad>'] + sorted(set([char for seq in seqs for char in seq]))
#
#
#
# ## Step 2: Load indexed data (list of instances, where each instance is list of character indices) ##
# ##-------------------------------------------------------------------------------------------------##
# vectorized_seqs = [[vocab.index(tok) for tok in seq]for seq in seqs]
#
# embed = Embedding(len(vocab), 4) # embedding_dim = 4
# lstm = LSTM(input_size=4, hidden_size=5, batch_first=True,bidirectional=False,num_layers=2) # input_dim = 4, hidden_dim = 5
#
#
#
#
# # get the length of each seq in your batch
# seq_lengths = LongTensor(list(map(len, vectorized_seqs)))
#
#
# seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
# # seq_tensor => [[0 0 0 0 0 0 0 0]
# #                [0 0 0 0 0 0 0 0]
# #                [0 0 0 0 0 0 0 0]]
#
# for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
#     seq_tensor[idx, :seqlen] = LongTensor(seq)
#
#
# seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
# _, un_idx = torch.sort(perm_idx, dim=0)
# seq_tensor = seq_tensor[perm_idx]
#
#
# embedded_seq_tensor = embed(seq_tensor)
#
#
# packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
#
#
# packed_output, state = lstm(packed_input)
#
# output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
#
#
# # print(ht)
# # (sat,cell) = state
# # print(sat)
# # print("/n")
# # print(cell)
# # print("/n")
# # print(state)
# # state = (state[0][::2], state[1][::2])
# # print("/n")
# # print(state)
# # print("/n")
# # output = torch.index_select(output, 0, un_idx)
# # # print(output[:,-1,:])
# # a = [7,3,5]
# #
# # print(a)
# # a= torch.tensor(a).unsqueeze(1)
# # print(a-1)
# # a = a.repeat(1,10)
# # # print(a
# # a = a.view(-1, 1,10)
# # # print(a.shape)
# # # print(a)
# # # print(output)
# # pooled_sequence_output = output.gather(                      # (B, H*)
# #             dim=1,
# #             index=a
# #         ).squeeze()
# # print(pooled_sequence_output)
# # hidden = torch.index_select(ht, 0, un_idx)
#
#
#
# # final = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1)
# # print(final)
# # final= torch.index_select(final, 0, un_idx)
# # print(final)
# # print(final.shape)
# # print(output)
# # print(pooled_sequence_output)
#
# # import torch
# # import warnings
# # warnings.filterwarnings('ignore')
#
# # import torch.nn as nn
# # import pandas as pd
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader
# # import train
# # import config
# # import dataloader
# # from lstm import LSTMBaseline
# # from earlystopping import EarlyStopping
# # import os
# # import ray
# # config.seed_torch()
# # from functools import partial
# # from ray import tune
# # from ray.tune import CLIReporter
# # from ray.tune.schedulers import ASHAScheduler
# # def objective(x, a, b):
# #     return a * (x ** 0.5) + b
# # def trainable(config):
# #     # config (dict): A dict of hyperparameters.
# #
# #     for x in range(20):
# #         intermediate_score = objective(x, config["a"], config["b"])
# #
# #         tune.report(score=intermediate_score)  # This sends the score to Tune.
# #
# # analysis = tune.run(
# #     trainable,
# #     config={"a": 2, "b": 4}
# # )
# #
# # print("best config: ", analysis.get_best_config(metric="score", mode="max"))
#
#
# #
# a = {"a":1,"b":2}
# for x, y in a.items():
#     a[x] = int(y)+1
# print(a)
# from transformers import BertModel,BertTokenizer
# import torch
# print(torch.__version__)
#
#
# BERT_PATH = '/Users/ximing/Desktop/bert-base-uncased'
#
# tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
#
# print(tokenizer.tokenize('I have a good time, thank you.'))
#
# bert = BertModel.from_pretrained(BERT_PATH)
#
# print('load bert model over')


# def Convert(string):
#     li = list(string.split(" "))
#     return li
#
#
# # Driver code
# str1 = "Geeks for Geeks"
# print(Convert(str1))
