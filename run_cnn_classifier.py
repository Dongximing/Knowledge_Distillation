import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from tqdm import tqdm
from utils import IMDB_indexing, pad_sequence
from models import CNN_Baseline
import csv
import pandas as pd
import argparse
import logging
import os
import pickle
import sys
def prepare_dateset(train_data_path, validation_data_path,test_data_path):
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
    training_review = training.Review
    training_sentiment = training.Sentiment

    for text,label in zip(training_review,training_sentiment):
        training_texts.append(text)
        training_labels.append(label)
    print("Finish loading training data")
    logging.info("Finish loading training data")

    # validation #
    print('Start loading validation data')
    logging.info("Start loading validation data")

    validation = pd.read_csv(validation_data_path)
    validation_review = validation.Review
    validation_sentiment = validation.Sentiment


    for text,label in zip(validation_review,validation_sentiment):
        validation_texts.append(text)
        validation_labels.append(label)
    print("Finish loading validation data")
    logging.info("Finish loading validation data")
    # testing #

    print('Start loading testing data')
    logging.info("Start loading testing data")

    testing = pd.read_csv(test_data_path)
    testing_review = testing.Review
    testing_sentiment = testing.Sentiment
    for text, label in zip(testing_review, testing_sentiment):
        testing_texts.append(text)
        testing_labels.append(label)
    print("Finish loading testing data")
    logging.info("Finish loading testing data")

    print('prepare training and test sets')
    logging.info('Prepare training and test sets')

    train_dataset, validation_dataset,test_dataset = IMDB_indexing(training_texts,training_labels,validation_texts,validation_labels, testing_texts,testing_labels)
    print('building vocab')
    logging.info('Build vocab')
    vocab = train_dataset.get_vocab()
    vocab_size = len(vocab)

    return train_dataset,validation_dataset,test_dataset, vocab,vocab_size

def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    # check if the dataset if train or test
    if len(batch[0]) == 2:
        label = [entry[0] for entry in batch]

        # padding according to the maximum sequence length in batch
        text = [entry[1] for entry in batch]
        text_length = [len(seq) for seq in text]
        text = pad_sequence(text, ksz=10, batch_first=True)
        return text, text_length, label
    else:
        text = [entry for entry in batch]
        text_length = [len(seq) for seq in text]
        text = pad_sequence(text, ksz=10, batch_first=True)
        return text, text_length
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(train_dataset,model,batch_sz,criterion,device,optimizer,lr_scheduler):
    training = DataLoader(train_dataset,collate_fn = generate_batch, batch_size=batch_sz,shuffle=True)
    for i,(text, length,label) in enumerate(training):
        text_length = torch.Tensor(length)
        text_length = text_length.to(device)
        text = text.to(device)
        label = label.to(device,torch.long)
        optimizer.zero_grad()
        output = model(text,text_length)
        loss = criterion(output,label)
        acc = categorical_accuracy(output, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    return epoch_loss / len(training), epoch_acc / len(training)


def validate(validation_dataset, model, batch_sz, criterion, device):
    validate = DataLoader(validation_dataset,collate_fn=generate_batch,batch_size=batch_sz,shuffle=False)
    for i,(text, length,label) in enumerate(validate):
        text_length = torch.Tensor(length)
        text_length = text_length.to(device)
        text = text.to(device)
        label = label.to(device,torch.long)
        with torch.no_grad():
            output = model(text, text_length)
        loss = criterion(output,label)
        acc = categorical_accuracy(output, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(validate), epoch_acc / len(validate)







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--validation_path')
    parser.add_argument('--test_path')
    parser.add_argument('--nKernel', type=int, default=64)
    parser.add_argument('--ksz', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--scheduler_step_sz', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.98)
    args = parser.parse_args()


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dataset, validation_dataset, test_dataset, vocab, vocab_size = prepare_dateset(args.train_path,args.validation_path,args.test_path)

    cnn_model =CNN_Baseline(vocab_size = vocab_size, nKernel = parser.nKernel,)
    cnn_model.to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=args.lr)



if __name__ == "__main__":
    main()