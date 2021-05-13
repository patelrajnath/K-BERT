import argparse
import os

from seqeval import metrics
from datautils.biluo_from_predictions import get_bio
from eval.myeval import f1_score_span, precision_score_span, recall_score_span

file_in = 'outputs/evaluation-lstm-crf/conll-2003/trun_gt.txt'
file_in_predictions = 'outputs/evaluation-lstm-crf/conll-2003/trun_base_luke.txt'
file_in_predictions = 'outputs/evaluation-lstm-crf/conll-2003/trun_lstm.txt'
file_in_predictions = 'outputs/evaluation-lstm-crf/conll-2003/comb_trun_luke_lstm.txt'

with open(file_in, mode="r", encoding="utf-8") as f_test, \
        open(file_in_predictions, mode="r", encoding="utf-8") as fin_predict:

    labels_predict_all = []
    labels_true_all = []

    for line_id, zip_line in enumerate(zip(f_test, fin_predict)):
        labels, predict = zip_line

        labels_predict = predict.split()
        labels_true = labels.split()

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
