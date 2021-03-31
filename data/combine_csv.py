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
df_unique.to_csv('combined_3/train_filtered_combined_3.csv', index=False)
