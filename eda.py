import pandas as pd
import csv
training_da = pd.read_csv('/home/dongxx/projects/def-parimala/dongxx/data/datrain.csv')
# training_da.columns = ['Lines','ID','Review','Review clean','Rating','Sentiment','Set']
training_da.drop(columns=['ID','Review clean','Rating','Set'],inplace=True, axis=1)
training_da.to_csv('/home/dongxx/projects/def-parimala/dongxx/data/datrain2.csv',encoding='utf-8')
# with open('/home/dongxx/projects/def-parimala/dongxx/data/datrain.csv', 'r') as infile, open('/home/dongxx/projects/def-parimala/dongxx/data/reordered.csv', 'a') as outfile:
#     # output dict needs a list for new column ordering
#     fieldnames = ['Sentiment','Review']
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     # reorder the header first
#     writer.writeheader()
#     for row in csv.DictReader(infile):
#         # writes the reordered rows to the new file
#         writer.writerow(row)