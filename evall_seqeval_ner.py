import argparse
import os

from seqeval import metrics
from create_word_predictions_with_voted_selection import tokenize_word, voting_choicer
from datautils.biluo_from_predictions import get_bio
from eval.myeval import f1_score_span, precision_score_span, recall_score_span

# file_in = 'outputs/evaluation-lstm-crf/conll-2003/eng.testa.dev.csv'
# file_in_predictions = 'outputs/evaluation-lstm-crf/conll-2003/test_avaluation_code_lstm_crf_predictions.txt'
file_in = 'data/combined_3/test_combined_3.csv'
# file_in_predictions = 'outputs/evaluation-lstm-crf/kaggle/kaggle_baseline_lstm_crf_predictions.txt'
# file_in_predictions = 'outputs/evaluation-lstm-crf/kaggle/kaggle_pretrain_lstm_crf_predictions.txt'
# file_in_predictions = 'outputs/evaluation-lstm-crf/kaggle/kaggle_lstm_crf_finetune_from_kaggle_predictions.txt'
file_in_predictions = 'outputs/evaluation-lstm-crf/kaggle/' \
                      'kaggle_lstm_crf_freeze_encoder_finetune_from_kaggle_predictions.txt'

# file_in = 'data/nlu/bio/nlu_test.csv'
# file_in_predictions = 'outputs/evaluation-lstm-crf/nlu/nlu_baseline_lstm_crf_predictions.txt'
# file_in_predictions = 'outputs/evaluation-lstm-crf/nlu/nlu_lstm_crf_finetune_from_kaggle_predictions.txt'


with open(file_in, mode="r", encoding="utf-8") as f_test, \
        open(file_in_predictions, mode="r", encoding="utf-8") as fin_predict:

    f_test.readline()
    labels_predict_all = []
    labels_true_all = []

    for line_id, zip_line in enumerate(zip(f_test, fin_predict)):
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
        num_predictions = len(labels_predict)
        if len(labels_predict) != len(labels_true):
            offset = 0
            final_labels_predict = []
            final_labels_true = []
            for w, lt in zip(words, labels_true):
                sub_tokens = tokenize_word(w)
                num_subwords = len(sub_tokens)
                if offset < num_predictions:
                    curr_predictions = labels_predict[offset: offset + num_subwords]
                    curr_final_label = voting_choicer(curr_predictions)
                    final_labels_predict.append(curr_final_label)
                    final_labels_true.append(lt)
                    offset += num_subwords
                else:
                    break
            if len(final_labels_true) != len(final_labels_predict):
                print('Error!')
                exit()
            labels_predict_all.append(final_labels_predict)
            labels_true_all.append(final_labels_true)
        else:
            labels_predict_all.append(labels_predict)
            labels_true_all.append(labels_true)

    true_labels_final = []
    predicted_labels_final = []
    for line_id, line in enumerate(zip(labels_true_all, labels_predict_all)):
        true_labels, pred_labels = line
        pred_labels = [p.replace('_', '-') for p in pred_labels]
        true_labels = [t.replace('_', '-') for t in true_labels]

        bio_tags_true = get_bio(true_labels)
        bio_tags_predicted = get_bio(pred_labels)

        if len(bio_tags_true) != len(bio_tags_predicted):
            # print(len(bio_tags_true))
            # print(len(bio_tags_predicted))
            print(bio_tags_true)
            print(bio_tags_predicted)
            print(line_id)

        true_labels_final.append(bio_tags_true)
        predicted_labels_final.append(bio_tags_predicted)
    results = dict(
        f1=metrics.f1_score(true_labels_final, predicted_labels_final),
        precision=metrics.precision_score(true_labels_final, predicted_labels_final),
        recall=metrics.recall_score(true_labels_final, predicted_labels_final),
        f1_span=f1_score_span(true_labels_final, predicted_labels_final),
        precision_span=precision_score_span(true_labels_final, predicted_labels_final),
        recall_span=recall_score_span(true_labels_final, predicted_labels_final),
    )
    print(results)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output_dir', type=str, default='outputs')
#     parser.add_argument('--t_labels', type=str, default="ubuntu_label.txt")
#     parser.add_argument('--p_labels', type=str, default="ubuntu_predict.txt")
#     parser.add_argument('--text', type=str, default="ubuntu_text.txt")
#     return vars(parser.parse_args())
#
#
# if __name__ == '__main__':
#     eval_(**parse_args())
