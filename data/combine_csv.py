import pandas as pd

df_conll = pd.read_csv('conll_2003/eng.train.train_filtered.csv', sep='\t', index_col=None, header=0)
df_conll = df_conll[['text', 'labels']]
df_twitter = pd.read_csv('tweeter_nlp/ner.txt.train_filtered.csv', sep='\t', index_col=None, header=0)
df_gmb = pd.read_csv('GMB/ner_filtered.csv', sep='\t', index_col=None, header=0)
df_kaggle = pd.read_csv('kaggle-ner/ner_filtered.csv', sep='\t', index_col=None, header=0)

df_combined = pd.concat([df_conll, df_gmb, df_twitter, df_kaggle], axis=0, ignore_index=True)
print(df_combined.count(axis=0))
df_unique = df_combined.drop_duplicates()
print(df_unique.count(axis=0))
df_unique.to_csv('conll_2003/eng.train.train_filtered_combined.csv', index=False)
