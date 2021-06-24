import pandas as pd
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
####from textblob import TextBlob
####from nltk.corpus import stopwords
from sklearn import model_selection
# import  nltk
plt.switch_backend('agg')
# nltk.download('stopwords')
train_data = pd.read_excel('/home/dongx34/lstm/IMDB-Movie-Reviews-50k-main/train_data_complete.xlsx')
testing_data = pd.read_excel('/home/dongx34/lstm/IMDB-Movie-Reviews-50k-main/test_data_complete.xlsx')
import config
# print(train_data.head())
SEED = 100
# stop = stopwords.words('english')
config.seed_torch()
def data_cleaning(train_data):
    train_data['Review'] = train_data['Review'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
    train_data['Review'] = train_data['Review'].str.replace('[^\w\s]','')
    # train_data['Reviews'] = train_data['Reviews'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
    # train_data['Reviews'] = train_data['Reviews'].apply(lambda x: str(TextBlob(x).correct()))
    train_data['Sentiment'] = train_data['Sentiment'].apply(lambda x:1 if x=="pos" else 0)
data_cleaning(train_data)
data_cleaning(testing_data)

train_data, valid_data = model_selection.train_test_split(train_data,test_size=0.2,random_state=SEED,stratify=train_data['Sentiment'].values)
train_data= train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
train_data.to_csv('train.csv', encoding='utf-8')
valid_data.to_csv('valid.csv', encoding='utf-8')
testing_data.to_csv('test.csv', encoding='utf-8')
# dd = pd.Series(valid_data['Sentiment']).value_counts()
# sns.barplot(x=np.array(['negative','positive']),y=dd.values)
# plt.show()
# plt.savefig("1.pdf")
# print(train_data.info())
# print(valid_data.info())
# print(testing_data.info())