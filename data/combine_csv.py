import numpy
import pandas as pd

# df_conll = pd.read_csv('conll_2003/eng.train.train_filtered.csv', sep='\t', index_col=None, header=0)
# df_conll = df_conll[['text', 'labels']]
df_twitter = pd.read_csv('tweeter_nlp/ner.txt.train_filtered.csv', sep='\t', index_col=None, header=0)
df_gmb = pd.read_csv('GMB/ner_filtered.csv', sep='\t', index_col=None, header=0)
df_kaggle = pd.read_csv('kaggle-ner/ner_filtered.csv', sep='\t', index_col=None, header=0)

df_combined = pd.concat([df_kaggle, df_gmb, df_twitter], axis=0, ignore_index=True)
print(df_combined.count(axis=0))
df_unique = df_combined.drop_duplicates()
df_test = df_unique
print(df_unique.count(axis=0))

msk = numpy.random.rand(len(df_unique)) <= 0.9

train_df = df_unique[msk]
test = df_unique[~msk]
print(train_df.shape)
print(test.shape)
msk = numpy.random.rand(len(train_df)) <= 0.95
train = train_df[msk]
dev = train_df[~msk]
print(train.shape)
print(dev.shape)

train.to_csv('combined_3/train_filtered_combined_3.csv', index=False)
test.to_csv('combined_3/test_filtered_combined_3.csv', index=False)
dev.to_csv('combined_3/dev_filtered_combined_3.csv', index=False)
