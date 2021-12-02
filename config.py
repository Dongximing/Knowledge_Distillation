import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
import random
import os
def seed_torch(seed = 100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    np.random.RandomState(seed)
OUTPUT_DIM =2
N_LAYERS =2
BIDIRECTIONAL = True
HIDDEN_DIM = 256
DROPOUT = 0.25
EPOCHS = 25
BASELINE_EPOCHS = 15

MODEL_KD_PATH ='/home/dongxx/projects/def-mercer/dongxx/project/LSTM-baseline/kd.pt'
MODEL_Base_PATH ='/home/dongxx/projects/def-parimala/dongxx/Model_parameter/baseline_bas.pt'
#    Teachedr_MODEL_PATH ='/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/bert.pt'
MODEL_CNN_PATH = '/home/dongxx/projects/def-mercer/dongxx/Model_parameter/cnn.pt'
BERT_PATH = '/home/dongxx/projects/def-parimala/dongxx/Model_parameter/new_bert.pt'
MODEL_Base_PATH_bu = '/home/dongxx/projects/def-mercer/dongxx/Model_parameter/baseline.pt'
BERT_ft_PATH = '/home/dongxx/projects/def-mercer/dongxx/project/Model_parameter/new_ft_bert.pt'
MODEL_Base_PATH_fk ='/home/dongxx/projects/def-parimala/dongxx/Model_parameter/fk.pt'
MODEL_CNN_PATH_kd ='/home/dongxx/projects/def-mercer/dongxx/Model_parameter/cnn_kd.pt'