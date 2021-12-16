import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from tqdm import tqdm
from utils import IMDB_indexing, pad_sequenc
from models import CNN_Baseline
import torchtext.vocab
from transformers import BertTokenizer, BertModel
import csv
import pandas as pd
import argparse
import logging
import os
import pickle
import sys
import config
config.seed_torch()
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe,Vocab,Vectors
from tqdm import tqdm
from utils import IMDB_indexing,pad_sequencing,IMDB_kd_indexing
from models import CNN_Baseline,LSTMBaseline,BERTGRUSentiment
from torch.nn.utils.rnn import pad_sequence

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
import torch.nn.functional as F

from collections import Counter
import time
import copy
def weight_matrix(vocab, vectors, dim=100):
    weight_matrix = np.zeros([len(vocab.itos), dim])
    for i, token in enumerate(vocab.stoi):
        try:
            weight_matrix[i] = vectors.__getitem__(token)
        except KeyError:
            weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
    return torch.from_numpy(weight_matrix)
def prepare_dateset(train_data_path, validation_data_path,test_data_path, vocab):
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
    labellist = list(testing.Sentiment)
    print('prepare training and test sets')
    logging.info('Prepare training and test sets')
    tokenize = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    train_dataset, validation_dataset,testing_dataset = IMDB_kd_indexing(training_texts,training_labels,validation_texts,validation_labels,testing_texts,testing_labels,tokenize,vocab= vocab)
    print('building vocab')






    return train_dataset,validation_dataset,testing_dataset,labellist

def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    # check if the dataset if train or test
    if len(batch[0]) == 4:
        label = [entry[0] for entry in batch]

        # padding according to the maximum sequence length in batch
        text = [entry[1] for entry in batch]
        text_length = [len(seq) for seq in text]
        text= pad_sequenc(text,ksz=10, batch_first=True)
        bert_id = [torch.tensor(entry[2]) for entry in batch]
        # print(bert_id)
        bert_id = pad_sequence(bert_id, batch_first=True)
        attention_mask = [torch.tensor(entry[3]) for entry in batch]
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        return text, text_length, label,bert_id,attention_mask
    else:
        text = [entry for entry in batch]
        text_length = [len(seq) for seq in text]
        text = pad_sequence(text, ksz=10, batch_first=True)
        return text, text_length
def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):

    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss
def categorical_accuracy(preds, y):

    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc, top_pred
# training,cnn_model,bert_model,criterion,device,optimizer,lr_scheduler
def train_kd_fc(data_loader, device, bert_model, model,optimizer, criterion,criterion_kd,scheduler):
    model.train()
    a = 0.5
    epoch_loss = 0
    epoch_acc = 0
    hard_loss = 0
    soft_loss = 0

    for bi,data in tqdm(enumerate(data_loader),total = len(data_loader)):
        text, text_length, label, bert_id, attention_mask = data
        text_length = torch.Tensor(text_length)
        label = torch.tensor(label, dtype=torch.long)
        ids = text.to(device, dtype=torch.long)
        label = label.to(device)
        bert_id = bert_id.to(device, dtype=torch.long)
        bert_mask = attention_mask.to(device, dtype=torch.long)

        lengths = text_length.to(device, dtype=torch.int)

        targets = label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        with torch.no_grad():
            bert_output = bert_model(bert_id,bert_mask)

        outputs = model(ids)
        loss_soft =criterion_kd(outputs,bert_output)
        loss_hard = criterion(outputs, targets)
        loss = loss_hard*a + (1-a)*loss_soft
        # loss = loss_fn_kd(outputs,label,bert_output,T=10,alpha=0.5)

        acc,_ = categorical_accuracy(outputs, targets)
        loss.backward()
        optimizer.step()
        soft_loss += loss_soft.item()
        hard_loss += loss_hard.item()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    scheduler.step()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader),hard_loss / len(data_loader), soft_loss/ len(data_loader)

# def train(train_dataset,model,bert_model,criterion,device,optimizer,lr_scheduler):
#     model.train()
#     epoch_loss = 0
#     epoch_acc = 0
#
#     total_pred = []
#     for i,(text, length,label) in enumerate(train_dataset):
#         text_length = torch.Tensor(length)
#         label = torch.tensor(label,dtype=torch.long,device=device)
#         text_length = text_length.to(device)
#         text = text.to(device)
#
#         optimizer.zero_grad()
#         output = model(text)
#         loss = criterion(output,label)
#         acc,_= categorical_accuracy(output, label)
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#
#         loss.backward()
#         optimizer.step()
#     lr_scheduler.step()
#
#
#     return epoch_loss / len(train_dataset), epoch_acc / len(train_dataset)


def validate(validation_dataset, model, criterion, device):
    model.eval()
    total_pred = []
    epoch_loss = 0
    epoch_acc = 0

    for i,data in enumerate(validation_dataset):
        text, text_length, label, bert_id, attention_mask = data

        text_length = torch.Tensor(text_length)
        text_length = text_length.to(device)
        text = text.to(device)
        label = torch.tensor(label, dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(text)
        loss = criterion(output,label)
        acc,pred = categorical_accuracy(output, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        total_pred.append(pred)
    flat_list = [item for sublist in total_pred for item in sublist]

    return epoch_loss / len(validation_dataset), epoch_acc / len(validation_dataset),flat_list







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='/home/dongxx/projects/def-mercer/dongxx/IMDB_data/train.csv')
    parser.add_argument('--validation_path', type=str,
                        default='/home/dongxx/projects/def-mercer/dongxx/IMDB_data/valid.csv')
    parser.add_argument('--test_path', type=str,
                        default='/home/dongxx/projects/def-mercer/dongxx/IMDB_data/test.csv')
    parser.add_argument('--nKernel', type=int, default=64)
    parser.add_argument('--ksz', type=list, default=[3,4,5])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--scheduler_step_sz', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--number_class', type=int, default=2)

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    counter2 = Counter({'<unk>': 400002, '<pad>': 400001})

    glove = Vectors(name='../glove.6B.100d.txt')
    f = open('../glove.6B.{}d.txt'.format(100), 'r')
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
    train_dataset, validation_dataset, test_dataset, labellist = prepare_dateset(args.train_path, args.validation_path,
                                                                                 args.test_path, vocab)
    # modelvocab_size,hidden_dim,n_layers,dropout,number_class,b
    # model
    cnn_model =CNN_Baseline(vocab_size = vocab_size, nKernel = args.nKernel, ksz = args.ksz,number_class = args.number_class)
    cnn_model.to(device)
    bert = BertModel.from_pretrained('bert-base-uncased')

    bert_model = BERTGRUSentiment(bert,
                                  config.HIDDEN_DIM,
                                  config.OUTPUT_DIM,
                                  config.N_LAYERS,
                                  config.BIDIRECTIONAL,
                                  config.DROPOUT)
    bert_model.load_state_dict(torch.load('/home/dongxx/projects/def-mercer/dongxx/project/Model_parameter/new_bert.pt'))
    bert_model.to(device)
    bert_model.eval()
    #opt scheduler criterion
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_gamma, step_size=10)
    criterion = nn.CrossEntropyLoss()

    kd_critertion = nn.MSELoss()
    kd_critertion = kd_critertion.to(device)
    criterion.to(device)

    training = DataLoader(train_dataset,collate_fn = generate_batch, batch_size=args.batch_sz,shuffle=True)
    validation = DataLoader(validation_dataset, collate_fn= generate_batch, batch_size=args.batch_sz, shuffle=False)
    testing = DataLoader(test_dataset, collate_fn=generate_batch, batch_size=args.batch_sz, shuffle=False)
    cnn_model.embedding_layer.weight.data.copy_(weight_matrix(vocab, glove)).to(device)
    cnn_model.embedding_layer.weight.data[1] = torch.zeros(100)
    cnn_model.embedding_layer.weight.data[0] = torch.zeros(100)
    cnn_model.embedding_layer.weight.requires_grad = False

    best_loss = float('inf')
    for epoch in range(args.num_epochs):

        train_loss, train_acc,soft_loss,hard_loss =train_kd_fc(training,device,bert_model,cnn_model,optimizer,criterion,kd_critertion,lr_scheduler)

        valid_loss, valid_acc,_ = validate(validation,cnn_model,criterion,device)
        print("epoch is ",epoch)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(cnn_model.state_dict(), config.MODEL_CNN_PATH_kd)
    print("training done")

    print("testing")
    cnn_model.load_state_dict(torch.load(config.MODEL_CNN_PATH_kd))
    test_loss, test_acc, flat_list = validate(testing, cnn_model, criterion, device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print("testing done")




if __name__ == "__main__":
    main()