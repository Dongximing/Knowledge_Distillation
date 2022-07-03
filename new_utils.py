import torchtext
import logging
import torch
#11111
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))




def _text_kd_iterator(text, labels=None, training_logits= None,ngrams=1, yield_label=False):
    tokenizer = get_tokenizer('basic_english')
    for i, bert_text in enumerate(text):
        # print(text)
        texts = tokenizer(bert_text)
        # filtered_text = [word for word in texts ]

        filtered_text = [word for word in texts if word not in stop_words ]
        # print(filtered_text)
        if yield_label:
            label = labels[i]
            logit = training_logits[i]
            yield label, logit, ngrams_iterator(filtered_text, ngrams)
        else:
            yield ngrams_iterator(filtered_text, ngrams)

def _create_data_kd_from_iterator(vocab,tokenizer, iterator, include_unk, is_test=False):
    data = []
    with tqdm(unit_scale=0, unit='lines') as t:
            for label, logit, text in iterator:

                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                           for token in text]))

                tokens = torch.tensor(token_ids)

                if len(tokens) == 0:
                    logging.info('Row contains no tokens.')
                data.append((label,tokens,logit))

                t.update(1)
            return data
# def _create_data_kd_prompt_from_iterator(vocab,tokenizer, iterator, include_unk, is_test=False):
#     data = []
#     with tqdm(unit_scale=0, unit='lines') as t:
#         if is_test:
#             for text in iterator:
#                 if include_unk:
#                     tokens = torch.tensor([vocab[token] for token in text])
#                 else:
#                     token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
#                                                                            for token in text]))
#                     tokens = torch.tensor(token_ids)
#                 if len(tokens) == 0:
#                     logging.info('Row contains no tokens.')
#                 data.append(tokens)
#                 t.update(1)
#             return data
#         else:
#             for label,example, text in iterator:
#                 if include_unk:
#                     print(text)
#                     tokens = torch.tensor([vocab[token] for token in text])
#
#                 else:
#
#
#                     # token_ids = list(filter(lambda x: x is not 0, [vocab[token]
#                     #                                                   for token in text]))
#                     token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
#                                                                            for token in text]))
#
#                     tokens = torch.tensor(token_ids)
#
#                     # print("tokens",tokens)
#                 if len(tokens) == 0:
#                     logging.info('Row contains no tokens.')
#                 data.append((label,tokens,example ))
#
#                 t.update(1)
#             return data
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, data):

        super(IMDBDataset, self).__init__()
        self._vocab = vocab
        self._data = data


    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_vocab(self):
        return self._vocab
def _setup_kd_datasets(train_text, train_labels, training_logits,validation_text, validation_labels, test_text,test_labels, tokenize,vocab,ngrams=1, include_unk=False):


    logging.info('Creating training data')
    train_data = _create_data_kd_from_iterator(
        vocab, tokenize,_text_kd_iterator(train_text,labels=train_labels,training_logits =training_logits, ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)
    logging.info('Creating validation data')
    validation_data =_create_data_kd_from_iterator(
        vocab, tokenize,_text_kd_iterator(validation_text, labels=validation_labels,training_logits =5000*[1.0], ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)

    logging.info('Creating testing data')
    test_data= _create_data_kd_from_iterator(
        vocab,tokenize, _text_kd_iterator(test_text, labels=test_labels, training_logits =25000*[1.0],ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)
    # logging.info('Total number of labels in training set:'.format(len(train_labels)))
    return (IMDBDataset(vocab, train_data),
            IMDBDataset(vocab,validation_data),
            IMDBDataset(vocab, test_data)

            )

def IMDB_kd_indexing(train_text, train_labels,training_logits, validation_text, validation_labels, test_text,test_labels,tokenize,vocab,ngrams=1, include_unk=False):

    return _setup_kd_datasets(train_text, train_labels, training_logits,validation_text, validation_labels, test_text, test_labels, tokenize,vocab,ngrams, include_unk)

def pad_sequenc(sequences, ksz, batch_first=False, padding_value=1):

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
def pad_sequencing(sequences, ksz, batch_first=False, padding_value=1):

    max_size = sequences[0].size()

    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    # print(max_len)
    if max_len > ksz:
        max_len = ksz
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims



    out_tensor = sequences[0].new_full(out_dims, padding_value)
    mask_tensor = sequences[0].new_full(out_dims, 0)

    true =[]
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # print(length)
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





