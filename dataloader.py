import torch
from torch.utils.data import Dataset

import config

config.seed_torch()
class IMDBDataset(Dataset):

        def __init__(self,review,target,word2id):
            self.review = review
            self.target = target
            self.wor2id = word2id
        def __len__(self):
            return len(self.review)

        def __getitem__(self, item):
            review = str(self.review[item])
            review = " ".join(review.split())
            review = review.split()
            review = convert(review)
            ids = word2id(review,self.wor2id)
            ids, length = pad_samples(ids,512,0)

            return {
                "text":torch.tensor(ids,dtype=torch.long),
                "target":torch.tensor(self.target[item],dtype=torch.float),
                "length":torch.tensor(length,dtype=torch.int)
            }
def convert(lst):
    # print(lst)
    return ([i for item in lst for i in item.split()])
def word2id(review,word2id):
    # print(review)
    ids =[]
    for index in range(len(review)):
        word = review[index]
        # print(word)
        if word in word2id:
           id = word2id[word]
        else:
            id = 0
            # print(word)

        ids.append(id)
    # print(ids)
    return ids
def pad_samples(feature, maxlen, PAD=0):

    if len(feature) >= maxlen:
        feature = feature[:maxlen]
        length = len(feature)

    else:

        length = len(feature)
        while(len(feature) < maxlen):
            feature.append(PAD)


    return feature,length






