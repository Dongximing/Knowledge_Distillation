import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import IMDB_indexing,bert_IMDB
from models import BERTGRUSentiment
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
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer,BertModel
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    label = [torch.tensor(entry['label'] for entry in batch)]
    return input_ids, attention_mask, label
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
    
    tokenizers = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    
    
    train_dataset, validation_dataset,testing_dataset = bert_IMDB(training_texts,training_labels,validation_texts,validation_labels,testing_texts,testing_labels,tokenizer=tokenizers, max_len=512)
    
    return train_dataset, validation_dataset, testing_dataset
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(train_dataset,model,criterion,device,optimizer,lr_scheduler):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    # if epoche>1:
    #     model.embedding_layer.weight.requires_grad = False


    for i,data in tqdm(enumerate(train_dataset),total = len(train_dataset)):
        input_ids, attention_mask, label = data
        input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(ids= input_ids, mask= attention_mask)
        loss = criterion(output,label)
        acc = categorical_accuracy(output, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    return epoch_loss / len(train_dataset), epoch_acc / len(train_dataset)


def validate(validation_dataset, model, criterion, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    for i,data in enumerate(validation_dataset):
        input_ids, attention_mask, label = data
        input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
        with torch.no_grad():
            output = model(ids = input_ids,attention_mask =attention_mask)
        loss = criterion(output,label)
        acc = categorical_accuracy(output, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(validation_dataset), epoch_acc / len(validation_dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='/home/dongxx/projects/def-mercer/dongxx/project/data/train.csv')
    parser.add_argument('--validation_path',type= str,default='/home/dongxx/projects/def-mercer/dongxx/project/data/valid.csv')
    parser.add_argument('--test_path',type= str,default='/home/dongxx/projects/def-mercer/dongxx/project/data/test.csv')

    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_sz', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--scheduler_step_sz', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--number_class', type=int, default=2)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert = BertModel.from_pretrained('bert-base-uncased')
    BertGRU_model = BERTGRUSentiment(bert,
                             config.HIDDEN_DIM,
                             config.OUTPUT_DIM,
                             config.N_LAYERS,
                             config.BIDIRECTIONAL,
                             config.DROPOUT)
    BertGRU_model.to(device)

    for name, param in BertGRU_model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False
    print(f'The model has {count_parameters(BertGRU_model):,} trainable parameters')
    optimizer = torch.optim.Adam(BertGRU_model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_gamma, step_size=5)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    train_dataset, validation_dataset,test_dataset = prepare_dateset(args.train_path, args.validation_path, args.test_path)


    training = DataLoader(train_dataset,collate_fn = generate_batch, batch_size=args.batch_sz,shuffle=True)
    validation = DataLoader(validation_dataset, collate_fn= generate_batch, batch_size=args.batch_sz, shuffle=False)
    testing = DataLoader(test_dataset, collate_fn= generate_batch, batch_size=args.batch_sz, shuffle=False)
    for epoch in range(args.num_epochs):
        start_time = time.time()
        # print("training emebedding")


        train_loss, train_acc = train(training,BertGRU_model,criterion,device,optimizer,lr_scheduler)
        # print("testing emebedding")
        valid_loss, valid_acc = validate(validation,BertGRU_model,criterion,device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(BertGRU_model.state_dict(), config.BERT)
    print("training done")

    print("testing")
    BertGRU_model.load_state_dict(torch.load(config.BERT))
    test_loss, test_acc = validate(testing,BertGRU_model,criterion,device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print("testing done")
if __name__ == "__main__":
    main()