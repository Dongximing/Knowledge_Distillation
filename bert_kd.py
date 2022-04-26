# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader
# from torchtext.vocab import GloVe,Vocab,Vectors
# from tqdm import tqdm
# from utils import IMDB_kd_indexing, pad_sequencing
# from models import CNN_Baseline,LSTMBaseline,BERTGRUSentiment,LSTM_atten,BERT
# import torch.nn.functional as F
#
# import torchtext.vocab
# import csv
# import pandas as pd
# import argparse
# import logging
# import os
# import pickle
# import sys
# import config
# config.seed_torch()
# from collections import Counter
# import time
# import copy
# from transformers import BertTokenizer, BertModel
# from torch.nn.utils.rnn import pad_sequence
# def loss_fn_kd(outputs, labels, teacher_outputs, T=10, alpha=0.5):
#
#     hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
#     soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
#                              F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
#     return hard_loss + soft_loss
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
#
# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs
# def weight_matrix(vocab, vectors, dim=100):
#     weight_matrix = np.zeros([len(vocab.itos), dim])
#     for i, token in enumerate(vocab.stoi):
#         # print(token)
#         # print(i)
#         try:
#             weight_matrix[i] = vectors.__getitem__(token)
#         except KeyError:
#             weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
#     return torch.from_numpy(weight_matrix)
# def prepare_dateset(train_data_path, validation_data_path,test_data_path,vocab):
#
#     training_texts = []
#     training_labels =[]
#     validation_texts = []
#     validation_labels = []
#     testing_texts = []
#     testing_labels = []
#     # training #
#     print('Start loading training data')
#     logging.info("Start loading training data")
#     training = pd.read_csv(train_data_path)
#
#     training_review = training.Review
#     training_sentiment = training.Sentiment
#
#     for text,label in zip(training_review,training_sentiment):
#         training_texts.append(text)
#         training_labels.append(label)
#     print("Finish loading training data")
#     logging.info("Finish loading training data")
#
#     # validation #
#     print('Start loading validation data')
#     logging.info("Start loading validation data")
#
#     validation = pd.read_csv(validation_data_path)
#     validation_review = validation.Review
#     validation_sentiment = validation.Sentiment
#
#
#     for text,label in zip(validation_review,validation_sentiment):
#         validation_texts.append(text)
#         validation_labels.append(label)
#     print("Finish loading validation data")
#     logging.info("Finish loading validation data")
#     # testing #
#
#     print('Start loading testing data')
#     logging.info("Start loading testing data")
#
#     testing = pd.read_csv(test_data_path)
#     testing_review = testing.Review
#     testing_sentiment = testing.Sentiment
#     for text, label in zip(testing_review, testing_sentiment):
#         testing_texts.append(text)
#         testing_labels.append(label)
#     print("Finish loading testing data")
#     logging.info("Finish loading testing data")
#
#     print('prepare training and test sets')
#     logging.info('Prepare training and test sets')
#     tokenizers = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#
#     train_dataset, validation_dataset,testing_dataset = IMDB_kd_indexing(training_texts,training_labels,validation_texts,validation_labels,testing_texts,testing_labels,tokenizers,vocab=vocab)
#     print('building vocab')
#
#
#     return train_dataset,validation_dataset,testing_dataset
#
# def generate_batch(batch):
#     """
#     Output:
#         text: the text entries in the data_batch are packed into a list and
#             concatenated as a single tensor for the input of nn.EmbeddingBag.
#         cls: a tensor saving the labels of individual text entries.
#     """
#     # check if the dataset if train or test
#     if len(batch[0]) == 5:
#         label = [entry[0] for entry in batch]
#
#         # padding according to the maximum sequence length in batch
#         text = [entry[1] for entry in batch]
#         # print(text)
#         # text_length = [len(seq) for seq in text]
#         # print(text_length)
#         text, text_length, mask= pad_sequencing(text, ksz = 512, batch_first=True)
#
#
#         bert_id = [torch.tensor(entry[2]) for entry in batch]
#         # print(bert_id)
#         bert_id = pad_sequence(bert_id, batch_first=True)
#         attention_mask = [torch.tensor(entry[3]) for entry in batch]
#         attention_mask = pad_sequence(attention_mask, batch_first=True)
#
#         token_type_ids = [torch.tensor(entry[4]) for entry in batch]
#         token_type_ids = pad_sequence(token_type_ids, batch_first=True)
#
#         return text, text_length, label,bert_id,attention_mask,token_type_ids,mask
#     else:
#         text = [entry for entry in batch]
#         text_length = [len(seq) for seq in text]
#         text = pad_sequence(text, ksz=10, batch_first=True)
#         return text, text_length
# def categorical_accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """
#     top_pred = preds.argmax(1, keepdim = True)
#     correct = top_pred.eq(y.view_as(top_pred)).sum()
#     acc = correct.float() / y.shape[0]
#     return acc
#
# def train_kd_fc(data_loader, device, bert_model, model,optimizer, criterion,criterion_kd,scheduler):
#     model.train()
#     a = 0.5
#     epoch_loss = 0
#     epoch_acc = 0
#
#
#     for bi,data in tqdm(enumerate(data_loader),total = len(data_loader)):
#         text, text_length, label, bert_id, attention_mask,token_type_ids,mask = data
#         text_length = torch.Tensor(text_length)
#         label = torch.tensor(label, dtype=torch.long)
#         ids = text.to(device, dtype=torch.long)
#         bert_id = bert_id.to(device, dtype=torch.long)
#         bert_mask = attention_mask.to(device, dtype=torch.long)
#
#         mask = mask.to(device)
#
#         lengths = text_length.to(device, dtype=torch.int)
#
#         targets = label.to(device, dtype=torch.long)
#         label = label.to(device)
#         token_type_ids = token_type_ids.to(device)
#         optimizer.zero_grad()
#         with torch.no_grad():
#             bert_output = bert_model(bert_id,bert_mask)
#
#         outputs = model(ids,lengths,mask)
#         loss_soft =criterion_kd(outputs,bert_output)
#         loss_hard = criterion(outputs, targets)
#         loss = loss_hard*a + (1-a)*loss_soft
#         # loss = loss_fn_kd(outputs,label,bert_output,T=10,alpha=0.5)
#
#         acc = categorical_accuracy(outputs, targets)
#         loss.backward()
#         #   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         # soft_loss += loss_soft.item()
#         # hard_loss += loss_hard.item()
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#     scheduler.step()
#     return epoch_loss / len(data_loader), epoch_acc / len(data_loader)
#         #,hard_loss / len(data_loader), soft_loss/ len(data_loader)
#
#
# def validate(validation_dataset, model, criterion, device):
#
#     epoch_loss = 0
#     epoch_acc = 0
#
#     for i,data in enumerate(validation_dataset):
#         text, text_length, label, _, _ ,token_type_ids,mask= data
#         text_length = torch.Tensor(text_length)
#         label = torch.tensor(label, dtype=torch.long)
#         text = text.to(device, dtype=torch.long)
#         mask = mask.to(device)
#
#
#         text_length = text_length.to(device,dtype=torch.int)
#
#         label = label.to(device)
#
#         with torch.no_grad():
#             output = model(text,text_length,mask)
#         loss = criterion(output,label)
#         acc = categorical_accuracy(output, label)
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#     return epoch_loss / len(validation_dataset), epoch_acc / len(validation_dataset)
#
#
#
#
#
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/data/train.csv')
#     parser.add_argument('--validation_path',type= str,default='/home/dongxx/projects/def-parimala/dongxx/data/valid.csv')
#     parser.add_argument('--test_path',type= str,default='/home/dongxx/projects/def-parimala/dongxx/data/test.csv')
#
#     parser.add_argument('--dropout', type=float, default=0.25)
#     parser.add_argument('--embedding_dim', type=int, default=100)
#     parser.add_argument('--num_epochs', type=int, default =25)
#     parser.add_argument('--batch_sz', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=1e-3)
#
#     parser.add_argument('--weight_decay', type=float, default=0.5)
#     parser.add_argument('--scheduler_step_sz', type=int, default=6)
#     parser.add_argument('--lr_gamma', type=float, default=0.1)
#     parser.add_argument('--number_class', type=int, default=2)
#
#     args = parser.parse_args()
#
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     counter2 = Counter({'<unk>': 400002, '<pad>': 400001})
#     glove = Vectors(name='/home/dongxx/projects/def-parimala/dongxx/glove.6B.100d.txt')
#     f = open('/home/dongxx/projects/def-parimala/dongxx/glove.6B.{}d.txt'.format(100), 'r')
#     loop = tqdm(f)
#     vob = {}
#     loop.set_description('Load Glove')
#     for i, line in enumerate(loop):
#         values = line.split()
#         word = values[0]
#         vob[word] = 400000 - i
#     counter1 = copy.deepcopy(vob)
#     f.close()
#     # counter1.update(counter2)
#     vocab = Vocab(counter1)
#     vocab_size = vocab.__len__()
#     print("vocab_size:", vocab_size)
#     train_dataset, validation_dataset,test_dataset = prepare_dateset(args.train_path, args.validation_path, args.test_path, vocab)
#     # modelvocab_size,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding_dim =10
#     LSTM_atten_model =LSTM_atten(vocab_size = vocab_size,hidden_dim = config.HIDDEN_DIM, n_layers =2, dropout = 0.25, number_class = args.number_class, bidirectional = True, embedding_dim =100)
#     LSTM_atten_model.to(device)
#     #opt scheduler criterion
#     optimizer = torch.optim.Adam(LSTM_atten_model.parameters(), lr=args.lr)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_gamma, step_size=10)
#     criterion = nn.CrossEntropyLoss()
#     kd_critertion = nn.MSELoss()
#     kd_critertion = kd_critertion.to(device)
#     # bert = Bert Model.from_pretrained('bert-base-uncased')
#     criterion = criterion.to(device)
#     bert = BertModel.from_pretrained('bert-base-uncased')
#     # bert_model = BERT(bert)
#     # bert_model.to(device)
#     bert_model = BERTGRUSentiment(bert,
#                                   config.HIDDEN_DIM,
#                                   config.OUTPUT_DIM,
#                                   config.N_LAYERS,
#                                   config.BIDIRECTIONAL,
#                                   config.DROPOUT)
#     bert_model.load_state_dict(torch.load(config.BERT_PATH))
#     bert_model.to(device)
#     bert_model.eval()
#
#
#     training = DataLoader(train_dataset,collate_fn = generate_batch, batch_size=args.batch_sz,shuffle=True)
#     validation = DataLoader(validation_dataset, collate_fn= generate_batch, batch_size=args.batch_sz, shuffle=False)
#     testing = DataLoader(test_dataset, collate_fn= generate_batch, batch_size=args.batch_sz, shuffle=False)
#
#
#
#
#     LSTM_atten_model.embedding_layer.weight.data.copy_(weight_matrix(vocab,glove)).to(device)
#     LSTM_atten_model.embedding_layer.weight.data[1] = torch.zeros(100)
#     LSTM_atten_model.embedding_layer.weight.data[0] = torch.zeros(100)
#
#
#     LSTM_atten_model.embedding_layer.weight.requires_grad = False
#     print(f'The LSTM KD attention model has {count_parameters(LSTM_atten_model):,} trainable parameters')
#
#     best_loss = float('inf')
#     print("training")
#     for epoch in range(args.num_epochs):
#         start_time = time.time()
#
#
#
#         train_loss, train_acc  = train_kd_fc(training, device, bert_model,LSTM_atten_model,optimizer, criterion,kd_critertion,lr_scheduler)
#
#         valid_loss, valid_acc = validate(validation,LSTM_atten_model,criterion,device)
#         end_time = time.time()
#         epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#         print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
#         print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
#         print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
#         # print("hard", hard)
#         # print("soft", soft)
#
#         if valid_loss < best_loss:
#             best_loss = valid_loss
#             torch.save(LSTM_atten_model.state_dict(), '/home/dongxx/projects/def-parimala/dongxx/Model_parameter/kd_atten.pt')
#     print("training done")
#
#     print("testing")
#     LSTM_atten_model.load_state_dict(torch.load('/home/dongxx/projects/def-parimala/dongxx/Model_parameter/kd_atten.pt'))
#     test_loss, test_acc = validate(testing,LSTM_atten_model,criterion,device)
#
#     print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
#     print("testing done")
#
#
#
#
#
#
# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe,Vocab
from tqdm import tqdm
from utils import IMDB_kd_indexing, pad_sequencing
from models import CNN_Baseline,LSTMBaseline,BERTGRUSentiment,LSTM_atten
from torchtext.vocab import GloVe,Vocab,Vectors
from transformers import BertTokenizer, BertModel,BertForSequenceClassification

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
        try:
            weight_matrix[i] = vectors.__getitem__(token)
        except KeyError:
            weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
    return torch.from_numpy(weight_matrix)
def prepare_dateset(train_data_path, validation_data_path,test_data_path,vocab):

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
    labellist = list(testing.Sentiment)

    print('prepare training and test sets')
    logging.info('Prepare training and test sets')
    tokenize = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    train_dataset, validation_dataset,testing_dataset = IMDB_kd_indexing(training_texts,training_labels,validation_texts,validation_labels,testing_texts,testing_labels,tokenize,vocab=vocab)
    print('building vocab')


    return train_dataset,validation_dataset,testing_dataset,labellist

def generate_batch(batch):

    if len(batch[0]) == 4:
        label = [entry[0] for entry in batch]

        # padding according to the maximum sequence length in batch
        text = [entry[1] for entry in batch]

        text, text_length,mask = pad_sequencing(text, ksz = 512, batch_first=True)


        bert_id = [torch.tensor(entry[2]) for entry in batch]
        # print(bert_id)
        bert_id = pad_sequence(bert_id, batch_first=True)
        attention_mask = [torch.tensor(entry[3]) for entry in batch]
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        return text, text_length, label,bert_id,attention_mask,mask
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
    return acc,top_pred

def train_kd_fc(data_loader, device, bert_model, model,optimizer, criterion,criterion_kd,scheduler):
    model.train()
    a = 0.5
    epoch_loss = 0
    epoch_acc = 0
    hard_loss = 0
    soft_loss = 0

    for bi,data in tqdm(enumerate(data_loader),total = len(data_loader)):
        text, text_length, label, bert_id, attention_mask,mask = data
        text_length = torch.Tensor(text_length)
        label = torch.tensor(label, dtype=torch.long)
        ids = text.to(device, dtype=torch.long)
        bert_id = bert_id.to(device, dtype=torch.long)
        bert_mask = attention_mask.to(device, dtype=torch.long)
        mask = mask.to(device)

        lengths = text_length.to(device, dtype=torch.int)

        targets = label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        inputs = {'input_ids': bert_id, 'attention_mask': bert_mask, 'labels': targets}
        optimizer.zero_grad()
        with torch.no_grad():
            bert_output = bert_model(**inputs)
            bert_logits = bert_output[1]


        outputs = model(ids,lengths,mask)
        loss_soft =criterion_kd(outputs,bert_logits)
        loss_hard = criterion(outputs, targets)
        loss = loss_hard*a + (1-a)*loss_soft
        acc,_ = categorical_accuracy(outputs, targets)
        loss.backward()
        #   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        soft_loss += loss_soft.item()
        hard_loss += loss_hard.item()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    scheduler.step()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader) ,hard_loss / len(data_loader), soft_loss/ len(data_loader)


def validate(validation_dataset, model, criterion, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    total_pred = []
    for i,data in enumerate(validation_dataset):
        text, text_length, label, _, _,mask = data
        text_length = torch.Tensor(text_length)
        label = torch.tensor(label, dtype=torch.long)
        text = text.to(device, dtype=torch.long)

        text_length = text_length.to(device,dtype=torch.int)
        mask = mask.to(device)


        label = label.to(device)

        with torch.no_grad():
            output = model(text,text_length,mask)
        loss = criterion(output,label)
        acc,pred = categorical_accuracy(output, label)
        epoch_loss += loss.item()
        total_pred.append(pred)
        epoch_acc += acc.item()
    flat_list = [item for sublist in total_pred for item in sublist]
    return epoch_loss / len(validation_dataset), epoch_acc / len(validation_dataset),flat_list







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/data/train.csv')
    parser.add_argument('--validation_path',type= str,default='/home/dongxx/projects/def-parimala/dongxx/data/valid.csv')
    parser.add_argument('--test_path',type= str,default='/home/dongxx/projects/def-parimala/dongxx/data/test.csv')

    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--scheduler_step_sz', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--number_class', type=int, default=2)

    args = parser.parse_args()

    # device
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
    glove = torchtext.vocab.GloVe(name='6B', dim=100,)
    # print(glove.get_vecs_by_tokens(['picture']))
    counter2 = Counter({'<unk>': 400000, '<pad>': 400001,'the':1})
    counter1 =  copy.deepcopy(glove.stoi)

    counter1.update(counter2)
    # print(counter1)

    vocab = Vocab(counter1)

    vocab_size=vocab.__len__()
    print("vocab_size:",vocab_size)
    # print(vocab.stoi)
    #
    # print(vocab.itos[2])
    # train_dataset, validation_dataset, test_dataset, vocab, vocab_size = prepare_dateset(args.train_path,args.validation_path)
    train_dataset, validation_dataset,test_dataset,labellist = prepare_dateset(args.train_path, args.validation_path, args.test_path, vocab)
    # modelvocab_size,hidden_dim,n_layers,dropout,number_class,bidirectional,embedding_dim =10
    LSTM_atten_model =LSTM_atten(vocab_size = vocab_size,hidden_dim = config.HIDDEN_DIM, n_layers =config.N_LAYERS, dropout = args.dropout, number_class = args.number_class, bidirectional = True, embedding_dim =100)
    LSTM_atten_model.to(device)
    #opt scheduler criterion
    optimizer = torch.optim.Adam(LSTM_atten_model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_gamma, step_size=10)
    criterion = nn.CrossEntropyLoss()
    kd_critertion = nn.MSELoss()
    kd_critertion = kd_critertion.to(device)

    criterion = criterion.to(device)
    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False)
    bert_model.load_state_dict(torch.load(config.BERT_nft_PATH))
    bert_model.to(device)
    bert_model.eval()



    training = DataLoader(train_dataset,collate_fn = generate_batch, batch_size=args.batch_sz,shuffle=True)
    validation = DataLoader(validation_dataset, collate_fn= generate_batch, batch_size=args.batch_sz, shuffle=False)
    testing = DataLoader(test_dataset, collate_fn= generate_batch, batch_size=args.batch_sz, shuffle=False)
    #loading vocab




    LSTM_atten_model.embedding_layer.weight.data.copy_(weight_matrix(vocab,glove)).to(device)
    LSTM_atten_model.embedding_layer.weight.data[1] = torch.zeros(100)
    LSTM_atten_model.embedding_layer.weight.data[0] = torch.zeros(100)


    LSTM_atten_model.embedding_layer.weight.requires_grad = False
    print(f'The lstm atten model model has {count_parameters(LSTM_atten_model):,} trainable parameters')

    best_loss = float('inf')
    print("training")
    for epoch in range(25):
        start_time = time.time()



        train_loss, train_acc,hard, soft  = train_kd_fc(training, device, bert_model,LSTM_atten_model,optimizer, criterion,kd_critertion,lr_scheduler)

        valid_loss, valid_acc,_ = validate(validation,LSTM_atten_model,criterion,device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print("hard", hard)
        print("soft", soft)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(LSTM_atten_model.state_dict(), '/home/dongxx/projects/def-parimala/dongxx/Model_parameter/rkd_atten.pt')
    print("training done")

    print("testing")
    LSTM_atten_model.load_state_dict(torch.load('/home/dongxx/projects/def-parimala/dongxx/Model_parameter/rkd_atten.pt'))
    test_loss, test_acc,flat_list = validate(testing,LSTM_atten_model,criterion,device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print("testing done")





if __name__ == "__main__":
    main()