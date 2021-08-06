import torchtext
import logging
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def _text_iterator(text, labels=None, ngrams=1, yield_label=False,remove_stopwords =False):
    tokenizer = get_tokenizer('basic_english')
    for i, text in enumerate(text):
        texts = tokenizer(text)
        if remove_stopwords:
            filtered_text = [word for word in texts if word not in stop_words]
        if yield_label:
            label = labels[i]
            yield label, ngrams_iterator(filtered_text, ngrams)
        else:
            yield ngrams_iterator(filtered_text, ngrams)
def _create_data_from_iterator(vocab, iterator, include_unk, is_test=False):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        if is_test:
            for text in iterator:
                if include_unk:
                    tokens = torch.tensor([vocab[token] for token in text])
                else:
                    token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                           for token in text]))
                    tokens = torch.tensor(token_ids)
                if len(tokens) == 0:
                    logging.info('Row contains no tokens.')
                data.append(tokens)
                t.update(1)
            return data
        else:
            for label, text in iterator:
                if include_unk:
                    tokens = torch.tensor([vocab[token] for token in text])
                else:
                    token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                           for token in text]))
                    tokens = torch.tensor(token_ids)
                if len(tokens) == 0:
                    logging.info('Row contains no tokens.')
                data.append((label, tokens))
                labels.extend(label)
                t.update(1)
            return data, set(labels)
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
def _setup_datasets(train_text, train_labels, validation_text, validation_labels, test_text, test_labels, ngrams=1, vocab=None, include_unk=False):
    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_text))
        vocab = build_vocab_from_iterator(_text_iterator(train_text, train_labels, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _text_iterator(train_text, labels=train_labels, ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)
    logging.info('Creating validation data')
    validation_data =_create_data_from_iterator(
        vocab, _text_iterator(validation_text, labels=validation_labels, ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)

    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _text_iterator(test_text, labels=test_labels, ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)
    logging.info('Total number of labels in training set:'.format(len(train_labels)))
    return (IMDBDataset(vocab, train_data),
            IMDBDataset(vocab,validation_data),
            IMDBDataset(vocab, test_data))

def IMDB_indexing(train_text, train_labels, validation_text, validation_labels,test_text, test_labels, ngrams=1, vocab=None, include_unk=False):


    return _setup_datasets(train_text, train_labels, validation_text, validation_labels,test_text, test_labels, ngrams, vocab, include_unk)
def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):
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