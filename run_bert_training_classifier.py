import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import IMDB_indexing, bert_IMDB,bert_ft_IMDB
from models import BERTGRUSentiment,BERT
import csv
import pandas as pd
import argparse
import logging
import os
import pickle
import sys
import config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

config.seed_torch()
from collections import Counter
import time
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertModel


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
    token_type_ids = [torch.tensor(entry['token_type_ids']) for entry in batch]
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    label = [entry['label'] for entry in batch]

    return input_ids, attention_mask,token_type_ids, label


def prepare_dateset(train_data_path, validation_data_path, test_data_path):

    training_texts = []
    training_labels = []
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

    for text, label in zip(training_review, training_sentiment):
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

    for text, label in zip(validation_review, validation_sentiment):
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
    labellist = list(testing.Sentiment)

    tokenizers = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_dataset, validation_dataset, testing_dataset = bert_ft_IMDB(training_texts, training_labels, validation_texts,
                                                                   validation_labels, testing_texts, testing_labels,
                                                                   tokenizer=tokenizers, max_len=512)

    return train_dataset, validation_dataset, testing_dataset,labellist


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc,top_pred


def train(train_dataset, model, criterion, device, optimizer, scheduler):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    # if epoche>1:
    #     model.embedding_layer.weight.requires_grad = False

    for i, data in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        input_ids, attention_mask,token_type_ids, label = data
        input_ids, attention_mask, token_type_ids, label = input_ids.to(device), attention_mask.to(device),token_type_ids.to(device), torch.LongTensor(label)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(ids=input_ids, mask=attention_mask,token_type_ids =token_type_ids )
        loss = criterion(output, label)
        acc,_ = categorical_accuracy(output, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return epoch_loss / len(train_dataset), epoch_acc / len(train_dataset)


def validate(validation_dataset, model, criterion, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    total_pred = []

    for i, data in enumerate(validation_dataset):
        input_ids, attention_mask,token_type_ids, label = data
        input_ids, attention_mask, token_type_ids, label = input_ids.to(device), attention_mask.to(device),token_type_ids.to(device), torch.LongTensor(label)
        label = label.to(device)
        with torch.no_grad():
            output = model(ids=input_ids, mask=attention_mask ,token_type_ids = token_type_ids)
        loss = criterion(output, label)
        acc, pred = categorical_accuracy(output, label)
        total_pred.append(pred)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    flat_list = [item for sublist in total_pred for item in sublist]
    return epoch_loss / len(validation_dataset), epoch_acc / len(validation_dataset),flat_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='/home/dongxx/projects/def-parimala/dongxx/data/train.csv')
    parser.add_argument('--validation_path', type=str,
                        default='/home/dongxx/projects/def-parimala/dongxx/data/valid.csv')
    parser.add_argument('--test_path', type=str,
                        default='/home/dongxx/projects/def-parimala/dongxx/data/test.csv')



    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_sz', type=int, default=8)


    parser.add_argument('--number_class', type=int, default=2)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert = BertModel.from_pretrained('bert-base-uncased')
    Bert_model = BERT(bert)
    Bert_model.to(device)
    param_optimizer = list(Bert_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=int(20000/8*5)
    )

    print(f'The Bert training model has {count_parameters(Bert_model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    train_dataset, validation_dataset, test_dataset,labellist = prepare_dateset(args.train_path, args.validation_path,
                                                                      args.test_path)

    training = DataLoader(train_dataset, collate_fn=generate_batch, batch_size=args.batch_sz, shuffle=True)
    validation = DataLoader(validation_dataset, collate_fn=generate_batch, batch_size=args.batch_sz, shuffle=False)
    testing = DataLoader(test_dataset, collate_fn=generate_batch, batch_size=args.batch_sz, shuffle=False)
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        start_time = time.time()
        # print("training emebedding")

        train_loss, train_acc = train(training, Bert_model, criterion, device, optimizer, scheduler)
        # print("testing emebedding")
        valid_loss, valid_acc,flat_list= validate(validation, Bert_model, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(Bert_model.state_dict(), config.BERT_ft_PATH)
    print("training done")

    print("testing")
    Bert_model.load_state_dict(torch.load(config.BERT_ft_PATH))
    test_loss, test_acc ,flat_list= validate(testing, Bert_model, criterion, device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print("testing done")


if __name__ == "__main__":
    main()