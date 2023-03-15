import config
import torch
import warnings
from lstm import LSTMBaseline
warnings.filterwarnings('ignore')
import torch.nn as nn
import pandas as pd
import dataloader
import train
import gensim
from torch.utils.data import Dataset
from gensim.test.utils import datapath,get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
config.seed_torch()
def loadword2vec(glove_dir,word2vec_dir):
    glove_input_file = datapath(glove_dir)
    word2vec_output_file = get_tmpfile(word2vec_dir)
    (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)

    return count, dimensions
glove_dir = '/home/dongx34/lstm/glove.6B.100d.txt'
word2vec_dir = '/home/dongx34/lstm/glove.6B.word2vec.100d.txt'
count,dimensions = loadword2vec(glove_dir,word2vec_dir)
#     print('count, dimensions', loadword2vec(glove_dir, word2vec_dir))
wvmodel = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.word2vec.100d.txt',
                                                        binary=False, encoding='utf-8')

word2id = {}
for i, word in enumerate(wvmodel.index_to_key):
    word2id[word] = i+1

w2v = torch.FloatTensor(wvmodel.vectors)
pad_unk = torch.zeros(([1, 100]))
pad_unk = torch.FloatTensor(pad_unk)
embedding = torch.cat((pad_unk,w2v),dim =0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMBaseline(dimensions,config.HIDDEN_DIM,config.N_LAYERS,config.DROPOUT,config.OUTPUT_DIM,config.BIDIRECTIONAL,embedding)
model.load_state_dict(torch.load('lstm.pt'))
model.to(device)
test_dir ='/home/dongx34/lstm/test.csv'
test_dataset = pd.read_csv(test_dir)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

test_dataset = dataloader.IMDBDataset(test_dataset['Review'].values,test_dataset['Sentiment'].values,word2id)


test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle = False
    )
test_loss, test_acc = train.eval_fc(test_data_loader,model,device,criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
# def convert(lst):
#     print(lst)
#     return ([i for item in lst for i in item.split()])
# def predict_sentiment(model, sentence):
#     sentence = sentence.split()
#     sentence = convert(sentence)
#     print(sentence)
#
#     ids =dataloader.word2id(sentence,word2id)
#
#     tensor = torch.LongTensor(ids).to(device)
#     print(tensor)
#
#     length = [len(ids)]
#     tensor = tensor.unsqueeze(0)
#     length_tensor = torch.LongTensor(length).to(device)
#     result = torch.sigmoid(model(tensor,length_tensor))
#     return result.item()
#
# print(predict_sentiment(model, "This film is great"))