import argparse
import os

from seqeval.metrics import f1_score


def eval_(output_dir, t_labels, p_labels, text):
    with open(os.path.join(output_dir, t_labels), 'r') as t, \
            open(os.path.join(output_dir, p_labels), 'r') as p, \
            open(os.path.join(output_dir, text), 'r') as textf:
        true_labels_all = []
        predicted_labels_all = []
        for text, true_labels, predicted_labels in zip(textf, t, p):
            true_labels = true_labels.strip().replace('_', '-').split()
            predicted_labels = predicted_labels.strip().replace('_', '-').split()
            true_labels_all.append(true_labels)
            predicted_labels_all.append(predicted_labels)
        f1 = f1_score(true_labels_all, predicted_labels_all)
        print(f1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--t_labels', type=str, default="ubuntu_label.txt")
    parser.add_argument('--p_labels', type=str, default="ubuntu_predict.txt")
    parser.add_argument('--text', type=str, default="ubuntu_text.txt")
    return vars(parser.parse_args())


if __name__ == '__main__':
    eval_(**parse_args())
