# Load Luke model.
import unicodedata
from collections import Counter

from transformers import RobertaTokenizer

from luke import ModelArchive

model_archive = ModelArchive.load('D:\Downloads\luke_base_500k.tar.gz')
tokenizer = model_archive.tokenizer


def is_punctuation(char):
    # obtained from:
    # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def tokenize_word(text):
    if (
            isinstance(tokenizer, RobertaTokenizer)
            and (text[0] != "'")
            and (len(text) != 1 or not is_punctuation(text))
    ):
        return tokenizer.tokenize(text, add_prefix_space=True)
    return tokenizer.tokenize(text)


def voting_choicer(items):
    votes = []
    joiner = '-'
    for item in items:
        if item and item != '[ENT]' and item != '[X]' and item != '[PAD]':
            if item == 'O' or item == '[CLS]' or item == '[SEP]':
                votes.append(item)
            else:
                joiner = item[1]
                votes.append(item[2:])

    vote_labels = Counter(votes)
    if not len(vote_labels):
        vote_labels = {'O': 1}
    lb = sorted(list(vote_labels), key=lambda x: vote_labels[x])

    final_lb = lb[-1]
    if final_lb == 'O' or final_lb == '[CLS]' or final_lb == '[SEP]':
        return final_lb
    else:
        return f'B{joiner}' + final_lb


if __name__ == '__main__':
    # file_in = 'data/nlu/nlu_test.csv'
    # file_in_predictions = 'outputs/evaluation/nlu/baseline-nlu_predictions_clean.txt'
    # file_out = 'outputs/evaluation/nlu/baseline-nlu_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/nlu/baseline-nlu_gold_clean_voted_selection.txt'

    # file_in = 'data/nlu/nlu_test_bio.csv'
    # file_in_predictions = 'outputs/evaluation/nlu/baseline-nlu-bio_predictions_clean.txt'

    # file_out = 'outputs/evaluation/nlu/baseline-nlu-bio_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/nlu/baseline-nlu-bio_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/nlu/baseline-nlu-bio_text_clean_voted_selection.txt'

    # file_in = 'data/nlu/nlu_test_bio.csv'
    # file_in_predictions = 'outputs/evaluation/nlu/finetune-stage2-nlu-bio_predictions_clean.txt'

    # file_out = 'outputs/evaluation/nlu/finetune-stage2-nlu-bio_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/nlu/finetune-stage2-nlu-bio_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/nlu/finetune-stage2-bio_text_clean_voted_selection.txt'

    # file_in = 'data/nlu/nlu_test.csv'
    # file_in_predictions = 'outputs/evaluation/nlu/finetune-stage2-nlu_predictions_clean.txt'

    # file_out = 'outputs/evaluation/nlu/finetune-stage2-nlu_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/nlu/finetune-stage2-nlu_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/nlu/finetune-stage2_text_clean_voted_selection.txt'

    # file_in = 'data/nlu/nlu_test.csv'
    # file_in_predictions = 'outputs/evaluation/nlu/baseline-nlu_predictions_clean.txt'

    # file_out = 'outputs/evaluation/nlu/baseline-nlu_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/nlu/baseline-nlu_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/nlu/baseline-nlu_text_clean_voted_selection.txt'

    # file_in = 'data/conll_2003/eng.testa.dev.csv'
    # file_in_predictions = 'outputs/evaluation/conll/baseline-conll_predictions_clean.txt'

    # file_out = 'outputs/evaluation/conll/baseline-conll_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/conll/baseline-conll_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/conll/baseline-conll_text_clean_voted_selection.txt'

    # file_in = 'data/conll_2003/eng.testa.dev.csv'
    # file_in_predictions = 'outputs/evaluation/conll/finetune-stage2-conll_from-conll_predictions_clean.txt'

    # file_out = 'outputs/evaluation/conll/finetune-stage2-conll_from-conll_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/conll/finetune-stage2-conll_from-conll_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/conll/finetune-stage2-conll_from-conll_text_clean_voted_selection.txt'

    # file_in = 'data/conll_2003/eng.testa.dev.csv'
    # file_in_predictions = 'outputs/evaluation/conll/finetune-stage2-conll_from-kaggle_predictions_clean.txt'

    # file_out = 'outputs/evaluation/conll/finetune-stage2-conll_from-kaggle_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/conll/finetune-stage2-conll_from-kaggle_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/conll/finetune-stage2-conll_from-kaggle_text_clean_voted_selection.txt'

    # file_in = 'data/combined_3/test_combined_3.csv'
    # file_in_predictions = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-conll_predictions_clean.txt'

    # file_out = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-conll_predictions_clean_voted_selection.txt'
    # file_out_true = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-conll_gold_clean_voted_selection.txt'
    # file_out_text = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-conll_text_clean_voted_selection.txt'

    file_in = 'data/combined_3/test_combined_3.csv'
    file_in_predictions = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-kaggle-pretrain-with-logits_predictions_clean.txt'

    file_out = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-kaggle-pretrain-with-logits_predictions_clean_voted_selection.txt'
    file_out_true = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-kaggle-pretrain-with-logits_gold_clean_voted_selection.txt'
    file_out_text = 'outputs/evaluation/kaggle/finetune-stage2-kaggle_from-kaggle-pretrain-with-logits_text_clean_voted_selection.txt'

    with open(file_in, mode="r", encoding="utf8") as f, \
            open(file_in_predictions, mode="r", encoding="utf8") as fin_predict, \
            open(file_out, mode="w", encoding="utf8") as fout, \
            open(file_out_true, mode="w", encoding="utf8") as fout_gold, \
            open(file_out_text, mode="w", encoding="utf8") as fout_text:
        f.readline()
        tokens, labels = [], []
        text_tokenized = []
        labels_predict_final = []

        for line_id, zip_line in enumerate(zip(f, fin_predict)):
            line, predict = zip_line

            tokens_subword = []
            fields = line.strip().split("\t")
            if len(fields) == 2:
                labels, tokens = fields
            elif len(fields) == 3:
                labels, tokens, cls = fields
            else:
                print(f'The data is not in accepted format at line no:{line_id}.. Ignored')
                continue
            labels_predict = predict.split()
            labels_true = labels.split()
            words = tokens.split()
            if len(labels_predict) != len(labels_true):
                offset = 0
                final_labels_predict = []
                for token in tokens.split():
                    sub_tokens = tokenize_word(token)
                    num_subwords = len(sub_tokens)
                    curr_predictions = labels_predict[offset: offset + num_subwords]
                    curr_final_label = voting_choicer(curr_predictions)
                    final_labels_predict.append(curr_final_label)
                    offset += num_subwords
                if len(words) != len(final_labels_predict):
                    print('Error!')
                    exit()
                fout.write(' '.join(final_labels_predict) + '\n')
            else:
                fout.write(' '.join(labels_predict) + '\n')
            fout_gold.write(' '.join(labels_true) + '\n')
            fout_text.write(tokens + '\n')
