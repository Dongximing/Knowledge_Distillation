import torch
import warnings
warnings.filterwarnings('ignore')
import gensim
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.test.utils import datapath,get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import train
import config
import dataloader
from lstm import LSTMBaseline
from earlystopping import EarlyStopping
import os 
config.seed_torch()
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def loadword2vec(glove_dir,word2vec_dir):
    glove_input_file = datapath(glove_dir)
    word2vec_output_file = get_tmpfile(word2vec_dir)
    (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
    return count, dimensions


   

def train_titanic(configs,checkpoint_dir=None,train_dir=None,valid_dir=None,glove_dir=None,word2vec_dir=None):
    # loading data
    train_dataset = pd.read_csv(train_dir)
    valid_dataset = pd.read_csv(valid_dir)


    count,dimensions = loadword2vec(glove_dir,word2vec_dir)
#     print('count, dimensions', loadword2vec(glove_dir, word2vec_dir))
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format('/home/dongxx/projects/def-mercer/dongxx/project/word2vec/glove.6B.word2vec.100d.txt',binary=False, encoding='utf-8')
    dimensions =100
    count = 400001


    word2id = {}
    for i, word in enumerate(wvmodel.index2word):
        word2id[word] = i+1

    w2v = torch.FloatTensor(wvmodel.vectors)
    pad_unk = torch.zeros(1, 100)
    pad_unk = torch.FloatTensor(pad_unk)
    embedding = torch.cat((pad_unk,w2v),dim =0)
    train_dataset = dataloader.IMDBDataset(train_dataset['Reviews'].values,train_dataset['Sentiment'].values,word2id)
    valid_dataset = dataloader.IMDBDataset(valid_dataset['Reviews'].values,valid_dataset['Sentiment'].values,word2id)


    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(configs["batch_size"]), shuffle = True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=int(configs["batch_size"]), shuffle = False
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMBaseline(dimensions,configs["hidden_dim"],config.N_LAYERS,config.DROPOUT,config.OUTPUT_DIM,config.BIDIRECTIONAL,embedding)

    model.to(device)



    optimizer = optim.Adam(model.parameters(),lr=configs["lr"])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
#     patience = 3
#     early_stopping = EarlyStopping(patience, verbose=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    best_loss = float('inf')

    for epoch in range (config.EPOCHS):

        train_loss, train_acc = train.train_fc(train_data_loader,device,model,optimizer,criterion,lr_scheduler)

        valid_loss, valid_acc = train.eval_fc(valid_data_loader,model,device,criterion)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        if valid_loss < best_loss:
            best_loss = valid_loss
            print(best_loss)
            torch.save(model.state_dict(), config.MODEL_PATH)


        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=valid_loss, accuracy=valid_acc)
#         early_stopping(valid_loss, model)

#         if early_stopping.early_stop:
#
#            print("Early stopping")
#            break


def main():

    train_dir = '/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/train.csv'
    valid_dir ='/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/valid.csv'
    glove_dir = '/home/dongxx/projects/def-mercer/dongxx/project/word2vec/glove.6B.100d.txt'
    word2vec_dir = '/home/dongxx/projects/def-mercer/dongxx/project/word2vec/glove.6B.word2vec.100d.txt'
    checkpoint_dir = config.MODEL_PATH
    max_num_epochs = 1
    num_samples = 1
    #
    configs = {
         "hidden_dim": tune.choice([64]),
         "lr" : tune.choice([1e-3]),
         "batch_size": tune.choice([128])

    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period =1 ,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["hidden_dim", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_titanic, checkpoint_dir=checkpoint_dir, train_dir=train_dir, valid_dir= valid_dir,glove_dir = glove_dir,word2vec_dir =word2vec_dir),

        resources_per_trial={"cpu": 4,"gpu":4},
        config=configs,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial(metric ="loss", mode ="min", scope ="all")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))




#


if __name__=="__main__":

    main()
