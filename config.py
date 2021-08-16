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
MODEL_Base_PATH ='/home/dongxx/projects/def-mercer/dongxx/project/LSTM-baseline/baseline.pt'
Teacher_MODEL_PATH ='/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/bert.pt'
MODEL_CNN_PATH = '/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/cnn.pt'
BERT_PATH =           '/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/new_bert.pt'

