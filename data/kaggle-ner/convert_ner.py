import json
import pandas as pd

data = pd.read_csv('ner_dataset.csv', sep='\t')
# data = pd.read_csv('ner_dataset.csv', encoding='windows-1252', error_bad_lines=False, sep='\t')
tags = list(set(data["Tag"].values.tolist()))

with open('labels.json', 'w', encoding='utf8') as f:
    json.dump(tags, f, indent=4)

data = data.fillna(method='ffill')

agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                   s["POS"].values.tolist(),
                                                   s["Tag"].values.tolist())]
grouped = data.groupby(["Sentence #"]).apply(agg_func)
sentences = [s for s in grouped]
X = [' '.join([w[0] for w in s]) for s in sentences]
y = [' '.join([w[2] for w in s]) for s in sentences]
data = zip(X, y)
df = pd.DataFrame(columns={'text', 'labels'}, data=data)
df.to_csv('ner.csv', index=False, encoding='utf-8', sep='\t')
