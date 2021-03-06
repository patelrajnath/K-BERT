import numpy
import pandas as pd
numpy.random.rand(4)

# df_conll = pd.read_csv('conll_2003/eng.train.train_filtered.csv', sep='\t', index_col=None, header=0)
# df_conll = df_conll[['text', 'labels']]
# df_twitter = pd.read_csv('tweeter_nlp/ner.txt.train_normalized.csv', sep='\t', index_col=None, header=0)
# df_gmb = pd.read_csv('GMB/ner_normalized.csv', sep='\t', index_col=None, header=0)
# df_kaggle = pd.read_csv('kaggle-ner/ner_normalized.csv', sep='\t', index_col=None, header=0)

# df_gmb = pd.read_csv('GMB/ner.csv', sep='\t', index_col=None, header=0)
# df_kaggle = pd.read_csv('kaggle-ner/ner.csv', sep=',', index_col=None, header=0)

df1 = pd.read_csv('combined_3/train_combined_3.csv', index=False, sep='\t')
df2 = pd.read_csv('combined_3/test_combined_3.csv', index=False, sep='\t')

df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)
print(df_combined.count(axis=0))
df_unique = df_combined.drop_duplicates()
df_test = df_unique
df_unique = df_unique[['labels', 'text']]
print(df_unique.count(axis=0))

dev = df_unique[:2000]
test = df_unique[2000:7000]
train = df_unique[7000:]
print(dev.shape)
print(test.shape)
print(train.shape)

# msk = numpy.random.rand(len(df_unique)) <= 0.9
#
# train_df = df_unique[msk]
# test = df_unique[~msk]
# print(train_df.shape)
# print(test.shape)
# msk = numpy.random.rand(len(train_df)) <= 0.95
# train = train_df[msk]
# dev = train_df[~msk]
# print(train.shape)
# print(dev.shape)

train.to_csv('combined_3/train_combined_3.csv', index=False, sep='\t')
# test.to_csv('combined_3/test_combined_3.csv', index=False, sep='\t')
# dev.to_csv('combined_3/dev_combined_3.csv', index=False, sep='\t')

# train.to_csv('combined_3/train_combined_3_normalized.csv', index=False, sep='\t')
# test.to_csv('combined_3/test_combined_3_normalized.csv', index=False, sep='\t')
# dev.to_csv('combined_3/dev_combined_3_normalized.csv', index=False, sep='\t')
