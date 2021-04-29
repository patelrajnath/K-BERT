from run_kg_luke_ner_lstm_crf import filter_kg_labels

t = ['O', 'B-PER', '[X]', '[ENT]', '[ENT]']
p = ['O', 'B-LOC', 'B-PER', 'O', 'O']
print(filter_kg_labels(t, p))

