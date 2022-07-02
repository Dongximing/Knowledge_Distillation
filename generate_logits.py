import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe,Vocab,Vectors
from tqdm import tqdm
from utils import IMDB_kd_indexing, pad_sequencing
from utils import IMDB_indexing, bert_IMDB
from models import CNN_Baseline,LSTMBaseline,BERTGRUSentiment
import torch.nn.functional as F
#123
import torchtext.vocab
import csv
import pandas as pd
import argparse
import logging
import os
import pickle
import sys
import config
config.seed_torch()
from collections import Counter
import time
import copy
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def weight_matrix(vocab, vectors, dim=100):
    weight_matrix = np.zeros([len(vocab.itos), dim])
    for i, token in enumerate(vocab.stoi):
        # print(token)
        # print(i)
        try:
            weight_matrix[i] = vectors.__getitem__(token)
        except KeyError:
            weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
    return torch.from_numpy(weight_matrix)
def prepare_dateset(train_data_path, validation_data_path,test_data_path,vocab):
    # with open(train_data_path,'r') as csvfile:
    #     csvreader = csv.reader(csvf
    training_texts = []
    training_labels =[]
    validation_texts = []
    validation_labels = []
    testing_texts = []
    testing_labels = []
    # training #
    print('Start loading training data')
    logging.info("Start loading training data")
    training = pd.read_csv(train_data_path)

    training_review = training.Review[:200]
    training_sentiment = training.Sentiment[:200]

    for text,label in zip(training_review,training_sentiment):
        training_texts.append(text)
        training_labels.append(label)
    print("Finish loading training data")
    logging.info("Finish loading training data")

    # validation #
    print('Start loading validation data')
    logging.info("Start loading validation data")

    validation = pd.read_csv(validation_data_path)
    validation_review = validation.Review[:200]
    validation_sentiment = validation.Sentiment[:200]


    for text,label in zip(validation_review,validation_sentiment):
        validation_texts.append(text)
        validation_labels.append(label)
    print("Finish loading validation data")
    logging.info("Finish loading validation data")
    # testing #

    print('Start loading testing data')
    logging.info("Start loading testing data")

    testing = pd.read_csv(test_data_path)
    testing_review = testing.Review[:200]
    testing_sentiment = testing.Sentiment[:200]
    for text, label in zip(testing_review, testing_sentiment):
        testing_texts.append(text)
        testing_labels.append(label)
    print("Finish loading testing data")
    logging.info("Finish loading testing data")

    print('prepare training and test sets')
    logging.info('Prepare training and test sets')
    tokenizers = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    train_dataset, validation_dataset,testing_dataset = bert_IMDB(training_texts, training_labels, validation_texts,
                                                                   validation_labels, testing_texts, testing_labels,
                                                                   tokenizer=tokenizers, max_len=512)
    print('building vocab')
    labellist = list(testing.Sentiment)

    # vocab = train_dataset.get_vocab()


    # vocab_size = len(vocab)
    # print('building vocab length',vocab_size)
    # logging.info(' Build vocab')

    return train_dataset,validation_dataset,testing_dataset,labellist

def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    input_ids = [torch.tensor(entry['input_ids']) for entry in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = [torch.tensor(entry['attention_mask']) for entry in batch]
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    label = [entry['label'] for entry in batch]

    return input_ids, attention_mask, label

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(data_loader, device, bert_model, criterion):
    bert_model.eval()
    epoch_loss = 0
    epoch_acc = 0
    result = []


    print(f'The model has {count_parameters(bert_model):,} trainable parameters')

    for bi,data in tqdm(enumerate(data_loader),total = len(data_loader)):
        input_ids, attention_mask, label = data
        input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), torch.LongTensor(label)

        label = label.to(device)
        with torch.no_grad():
            bert_output = bert_model(input_ids, attention_mask)
        result.append(bert_output)


        loss= criterion(bert_output,label)



        acc = categorical_accuracy(bert_output, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()




    return epoch_loss / len(data_loader), epoch_acc / len(data_loader),result










def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/data/train.csv')
    parser.add_argument('--validation_path',type= str,default='/home/dongxx/projects/def-parimala/dongxx/data/valid.csv')
    parser.add_argument('--test_path',type= str,default='/home/dongxx/projects/def-parimala/dongxx/data/test.csv')

    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--scheduler_step_sz', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--number_class', type=int, default=2)

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
    # glove = torchtext.vocab.GloVe(name='6B', dim=100,)
    # print(glove.get_vecs_by_tokens(['picture']))
    counter2 = Counter({'<unk>': 400002, '<pad>': 400001})
    glove = Vectors(name='/home/dongxx/projects/def-parimala/dongxx/glove.6B.100d.txt')
    f = open('/home/dongxx/projects/def-parimala/dongxx/glove.6B.{}d.txt'.format(100), 'r')
    loop = tqdm(f)
    vob = {}
    loop.set_description('Load Glove')
    for i, line in enumerate(loop):
        values = line.split()
        word = values[0]
        vob[word] = 400000 - i
    counter1 = copy.deepcopy(vob)
    f.close()

    counter1.update(counter2)
    vocab = Vocab(counter1)
    vocab_size = vocab.__len__()
    print("vocab_size:", vocab_size)

    # train_dataset, validation_dataset, test_dataset, vocab, vocab_size = prepare_dateset(args.train_path,args.validation_path)
    train_dataset, validation_dataset,test_dataset,labellist = prepare_dateset(args.train_path, args.validation_path, args.test_path, vocab)

    criterion = nn.CrossEntropyLoss()
    bert = BertModel.from_pretrained('bert-base-uncased')
    criterion = criterion.to(device)
    bert_model = BERTGRUSentiment(bert,
                                  config.HIDDEN_DIM,
                                  config.OUTPUT_DIM,
                                  config.N_LAYERS,
                                  config.BIDIRECTIONAL,
                                  config.DROPOUT)
    bert_model.load_state_dict(torch.load('/home/dongxx/projects/def-parimala/dongxx/Model_parameter/bert/new_bert.pt'))
    bert_model.to(device)


    training = DataLoader(train_dataset,collate_fn = generate_batch, batch_size=8,shuffle=False)






    best_loss = float('inf')
    print("training")
    for epoch in range(1):
        start_time = time.time()



        train_loss, train_acc,result  = train(training,device,bert_model,criterion)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')


        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    flat_list = [x for xs in result for x in xs]
    print(flat_list)
    df = pd.read_csv('/home/dongxx/projects/def-parimala/dongxx/data/train.csv')
    df['logit'] = flat_list
    df.to_csv('/home/dongxx/projects/def-parimala/dongxx/data/train.csv')









#

if __name__ == "__main__":
    main()