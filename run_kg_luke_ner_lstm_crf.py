# -*- encoding:utf -*-
"""
  This script provides an K-BERT example for NER.
"""
import argparse
import logging
import os
import sys

import seqeval
import torch
import torch.nn as nn

from collections import Counter

from seqeval.metrics import f1_score

from brain import config
from brain.knowgraph_english import KnowledgeGraph
from datautils.biluo_from_predictions import get_bio
from eval.myeval import f1_score_span, precision_score_span, recall_score_span
from luke import ModelArchive, LukeModel
from model_crf.decoders import NCRFDecoder, CRFDecoder
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import BertAdam
from uer.utils.constants import *
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from torch.nn import functional as F


fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_encoder(args, encoder, suffix='encoder'):
    try:
        os.makedirs(args.output_encoder)
    except:
        pass

    model_file = f"model_{suffix}.bin"
    torch.save(encoder.state_dict(), os.path.join(args.output_encoder, model_file))


def loss_fn(outputs, labels, mask):
    # the number of tokens is the sum of elements in mask
    num_labels = int(torch.sum(mask).item())

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels] * mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs) / num_labels


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


def filter_kg_labels(t, p, specials=('[ENT]', '[X]', '[PAD]')):
    t_filtered = []
    p_filtered = []
    for i, labels in enumerate(zip(t, p)):
        t_label, p_label = labels
        if t_label in specials:
            continue
        else:
            p_filtered.append(p_label)
            t_filtered.append(t_label)
    return t_filtered, p_filtered


class Batcher(object):
    def __init__(self, batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, segment_ids):
        self.batch_size = batch_size
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.mask_ids = mask_ids
        self.pos_ids = pos_ids
        self.vm_ids = vm_ids
        self.tag_ids = tag_ids
        self.segment_ids = segment_ids
        self.num_samples = self.input_ids.shape[0]
        self.indices = torch.randperm(self.num_samples)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= self.num_samples:
            self.indices = torch.randperm(self.num_samples)
            self.ptr = 0
            raise StopIteration
        else:
            batch_indices = self.indices[self.ptr: self.ptr + self.batch_size]
            self.ptr += self.batch_size

            return self.input_ids[batch_indices], self.label_ids[batch_indices], \
                   self.mask_ids[batch_indices], self.pos_ids[batch_indices], \
                   self.vm_ids[batch_indices], self.tag_ids[batch_indices], \
                   self.segment_ids[batch_indices]


class LukeTaggerMLP(nn.Module):
    def __init__(self, args, encoder):
        super(LukeTaggerMLP, self).__init__()
        self.args = args
        self.encoder = encoder
        self.labels_num = args.labels_num
        # Classification layer transforms the output to give the final output layer
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

        if self.args.freeze_encoder_weights:
            self.freeze()

    def luke_encode(self, word_ids, word_segment_ids, word_attention_mask, pos, vm):
        # Encoder.
        # print('word ids:', word_ids)
        word_sequence_output, pooled_output = self.encoder(word_ids, word_segment_ids=word_segment_ids,
                                                           word_attention_mask=word_attention_mask,
                                                           position_ids=pos, vm=vm)
        # run the LSTM along the sentences of length batch_max_len
        return word_sequence_output

    def get_logits(self, tensor):
        logits = self.output_layer(tensor)
        return logits

    def forward(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        tensor = self.luke_encode(word_ids, word_segment_ids, word_attention_mask, pos, vm)
        logits = self.get_logits(tensor)
        logits = logits.contiguous().view(-1, self.labels_num)
        outputs = F.log_softmax(logits, dim=-1)
        predict = outputs.argmax(dim=-1).view(-1, self.args.seq_length)
        return predict

    def score(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        tensor = self.luke_encode(word_ids, word_segment_ids, word_attention_mask, pos, vm)

        labels = labels.contiguous().view(-1)
        mask = (labels > 0).float().to(torch.device(labels.device))

        logits = self.get_logits(tensor)
        logits = logits.contiguous().view(-1, self.labels_num)
        outputs = F.log_softmax(logits, dim=-1)
        return loss_fn(outputs, labels, mask)

    def freeze(self):
        logger.info('The encoder has been frozen.')
        for param in self.encoder.parameters():
            param.requires_grad = False


class LukeTaggerLSTM(nn.Module):
    def __init__(self, args, encoder):
        super(LukeTaggerLSTM, self).__init__()
        self.encoder = encoder
        self.labels_num = args.labels_num
        self.lstm = nn.LSTM(args.emb_size, args.hidden_size // 2,
                            batch_first=True, bidirectional=True)
        # Classification layer transforms the output to give the final output layer
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

        if self.args.freeze_encoder_weights:
            self.freeze()

    def lstm_output(self, word_ids, word_segment_ids, word_attention_mask, pos, vm):
        # Encoder.
        # print('word ids:', word_ids)
        word_sequence_output, pooled_output = self.encoder(word_ids, word_segment_ids=word_segment_ids,
                                                           word_attention_mask=word_attention_mask,
                                                           position_ids=pos, vm=vm)
        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(word_sequence_output)  # dim: batch_size x batch_max_len x lstm_hidden_dim
        return tensor

    def get_logits(self, tensor):
        logits = self.output_layer(tensor)
        return logits

    def forward(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        tensor = self.lstm_output(word_ids, word_segment_ids, word_attention_mask, pos, vm)
        logits = self.get_logits(tensor)
        logits = logits.contiguous().view(-1, self.labels_num)
        outputs = F.log_softmax(logits, dim=-1)
        predict = outputs.argmax(dim=-1).view(-1, self.args.seq_length)
        return predict

    def score(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        labels = labels.contiguous().view(-1)
        mask = (labels > 0).float().to(torch.device(labels.device))

        tensor = self.lstm_output(word_ids, word_segment_ids, word_attention_mask, pos, vm)
        logits = self.get_logits(tensor)
        logits = logits.contiguous().view(-1, self.labels_num)
        outputs = F.log_softmax(logits, dim=-1)
        return loss_fn(outputs, labels, mask)

    def freeze(self):
        logger.info('The encoder has been frozen.')
        for param in self.encoder.parameters():
            param.requires_grad = False


class LukeTaggerLSTMCRF(nn.Module):
    def __init__(self, args, encoder):
        super(LukeTaggerLSTMCRF, self).__init__()
        self.encoder = encoder
        self.labels_num = args.labels_num
        self.lstm = nn.LSTM(args.emb_size, args.hidden_size // 2,
                            batch_first=True, bidirectional=True)
        # CRF layer transforms the output to give the final output layer
        self.crf = CRFDecoder.create(self.labels_num, args.hidden_size, args.device, args.seq_length)

        if self.args.freeze_encoder_weights:
            self.freeze()

    def lstm_output(self, word_ids, word_segment_ids, word_attention_mask, pos, vm):
        # Encoder.
        # print('word ids:', word_ids)
        word_sequence_output, pooled_output = self.encoder(word_ids, word_segment_ids=word_segment_ids,
                                                           word_attention_mask=word_attention_mask,
                                                           position_ids=pos, vm=vm)
        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(word_sequence_output)  # dim: batch_size x batch_max_len x lstm_hidden_dim
        return tensor

    def forward(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        labels_mask = (labels > 0).long()
        tensor = self.lstm_output(word_ids, word_segment_ids, word_attention_mask, pos, vm)
        return self.crf.forward(tensor, labels_mask)

    def score(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        labels_mask = (labels > 0).long()

        tensor = self.lstm_output(word_ids, word_segment_ids, word_attention_mask, pos, vm)
        return self.crf.score(tensor, labels_mask, labels)

    def freeze(self):
        logger.info('The encoder has been frozen.')
        for param in self.encoder.parameters():
            param.requires_grad = False


class LukeTaggerLSTMNCRF(nn.Module):
    def __init__(self, args, encoder):
        super(LukeTaggerLSTMNCRF, self).__init__()
        self.encoder = encoder
        self.labels_num = args.labels_num
        self.lstm = nn.LSTM(args.emb_size, args.hidden_size // 2,
                            batch_first=True, bidirectional=True)
        # CRF layer transforms the output to give the final output layer
        self.ncrf = NCRFDecoder.create(self.labels_num, args.hidden_size, args.device, args.seq_length)

        if self.args.freeze_encoder_weights:
            self.freeze()

    def lstm_output(self, word_ids, word_segment_ids, word_attention_mask, pos, vm):
        # Encoder.
        # print('word ids:', word_ids)
        word_sequence_output, pooled_output = self.encoder(word_ids, word_segment_ids=word_segment_ids,
                                                           word_attention_mask=word_attention_mask,
                                                           position_ids=pos, vm=vm)
        # run the LSTM along the sentences of length batch_max_len
        tensor, _ = self.lstm(word_sequence_output)  # dim: batch_size x batch_max_len x lstm_hidden_dim
        return tensor

    def forward(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        labels_mask = (labels > 0).long()
        tensor = self.lstm_output(word_ids, word_segment_ids, word_attention_mask, pos, vm)
        return self.ncrf.forward(tensor, labels_mask)

    def score(self, word_ids, word_segment_ids, word_attention_mask, labels, pos=None, vm=None, use_kg=True):
        if not use_kg:
            vm = None
        labels_mask = (labels > 0).long()

        tensor = self.lstm_output(word_ids, word_segment_ids, word_attention_mask, pos, vm)
        return self.ncrf.score(tensor, labels_mask, labels)

    def freeze(self):
        logger.info('The encoder has been frozen.')
        for param in self.encoder.parameters():
            param.requires_grad = False


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/tagger_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--output_encoder", default="./luke-models/", type=str,
                        help="Path of the output luke model.")
    parser.add_argument("--suffix_file_encoder", default="encoder", type=str,
                        help="output file suffix luke model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--output_file_prefix", type=str, required=True,
                        help="Prefix for file output.")
    parser.add_argument("--log_file", default='app.log')

    # Model options.
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=256, type=int,
                        help="Sequence length.")
    parser.add_argument("--classifier", choices=["mlp", "lstm", "lstm_crf", "lstm_ncrf"], default="mlp",
                        help="Classifier type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument('--freeze_encoder_weights', action='store_true', help="Enable to freeze the encoder weigths.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=2,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=35,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--use_kg", action='store_true', help="Enable the use of KG.")
    parser.add_argument("--dry_run", action='store_true', help="Dry run to test the implementation.")
    parser.add_argument("--voting_choicer", action='store_true',
                        help="Enable the Voting choicer to select the entity type.")
    parser.add_argument("--eval_kg_tag", action='store_true', help="Enable to include [ENT] tag in evaluation.")
    parser.add_argument("--use_subword_tag", action='store_true',
                        help="Enable to use separate tag for subword splits.")
    parser.add_argument("--debug", action='store_true', help="Enable debug.")
    parser.add_argument("--reverse_order", action='store_true', help="Reverse the feature selection order.")
    parser.add_argument("--max_entities", default=2, type=int,
                        help="Number of KG features.")
    parser.add_argument("--eval_range_with_types", action='store_true', help="Enable to eval range with types.")

    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    logging.basicConfig(filename=args.log_file, filemode='w', format=fmt)

    labels_map = {"[PAD]": 0, "[ENT]": 1, "[X]": 2, "[CLS]": 3, "[SEP]": 4}
    begin_ids = []

    # Find tagging labels
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            labels = line.strip().split("\t")[0].split()
            for l in labels:
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))
                    labels_map[l] = len(labels_map)

    idx_to_label = {labels_map[key]: key for key in labels_map}

    print(begin_ids)
    print("Labels: ", labels_map)
    args.labels_num = len(labels_map)

    # Build knowledge graph.
    if args.kg_name == 'none':
        kg_file = []
    else:
        kg_file = args.kg_name

    # Load Luke model.
    model_archive = ModelArchive.load(args.pretrained_model_path)
    tokenizer = model_archive.tokenizer

    # Handling space character in roberta tokenizer
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # Load the pretrained model
    encoder = LukeModel(model_archive.config)
    encoder.load_state_dict(model_archive.state_dict, strict=False)

    kg = KnowledgeGraph(kg_file=kg_file, tokenizer=tokenizer)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Build sequence labeling model.
    classifiers = {"mlp": LukeTaggerMLP,
                   "lstm": LukeTaggerLSTM,
                   "lstm_crf": LukeTaggerLSTMCRF,
                   "lstm_ncrf": LukeTaggerLSTMNCRF
                   }
    logger.info(f'The selected classifier is:{classifiers[args.classifier]}')
    model = classifiers[args.classifier](args, encoder)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, segment_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size, :]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vm_ids_batch = vm_ids[i * batch_size: (i + 1) * batch_size, :, :]
            tag_ids_batch = tag_ids[i * batch_size: (i + 1) * batch_size, :]
            segment_ids_batch = segment_ids[i * batch_size: (i + 1) * batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch, segment_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:, :]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            vm_ids_batch = vm_ids[instances_num // batch_size * batch_size:, :, :]
            tag_ids_batch = tag_ids[instances_num // batch_size * batch_size:, :]
            segment_ids_batch = segment_ids[instances_num // batch_size * batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch, segment_ids_batch

    # Read dataset.
    def read_dataset(path):
        dataset = []
        count = 0
        with open(path, mode="r", encoding="utf8") as f:
            f.readline()
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                fields = line.strip().split("\t")
                if len(fields) == 2:
                    labels, tokens = fields
                elif len(fields) == 3:
                    labels, tokens, cls = fields
                else:
                    print(f'The data is not in accepted format at line no:{line_id}.. Ignored')
                    continue

                tokens, pos, vm, tag = \
                    kg.add_knowledge_with_vm([tokens], [labels],
                                             use_kg=args.use_kg,
                                             max_length=args.seq_length,
                                             max_entities=args.max_entities,
                                             reverse_order=args.reverse_order)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]

                # tokens = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                non_pad_tokens = [tok for tok in tokens if tok != tokenizer.pad_token]
                num_tokens = len(non_pad_tokens)
                num_pad = len(tokens) - num_tokens

                labels = [config.CLS_TOKEN] + labels.split(" ") + [config.SEP_TOKEN]
                new_labels = []
                j = 0
                joiner = '-'
                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != tokenizer.pad_token:
                        cur_type = labels[j]
                        new_labels.append(cur_type)
                        if cur_type != 'O':
                            joiner = cur_type[1]
                            prev_label = cur_type[2:]
                        else:
                            prev_label = cur_type
                        j += 1
                    elif tag[i] == 1 and tokens[i] != tokenizer.pad_token:  # 是添加的实体
                        new_labels.append('[ENT]')
                    elif tag[i] == 2:
                        if prev_label == 'O':
                            new_labels.append('O')
                        else:
                            if args.use_subword_tag:
                                new_labels.append('[X]')
                            else:
                                new_labels.append(f'I{joiner}' + prev_label)
                    else:
                        new_labels.append(PAD_TOKEN)

                new_labels = [labels_map[l] for l in new_labels]

                # print(tokens)
                # print(labels)
                # print(tag)

                mask = [1] * (num_tokens) + [0] * num_pad
                word_segment_ids = [0] * (len(tokens))

                # print(len(tokens))
                # print(len(tag))
                # exit()
                # print(tokenizer.pad_token_id)

                # for i in range(len(tokens)):
                #     if tag[i] == 0 and tokens[i] != tokenizer.pad_token:
                #         new_labels.append(labels[j])
                #         j += 1
                #     elif tag[i] == 1 and tokens[i] != tokenizer.pad_token:  # 是添加的实体
                #         new_labels.append(labels_map['[ENT]'])
                #     elif tag[i] == 2:
                #         if args.use_subword_tag:
                #             new_labels.append(labels_map['[X]'])
                #         else:
                #             new_labels.append(labels_map['[ENT]'])
                #     else:
                #         new_labels.append(labels_map[PAD_TOKEN])

                # print(labels)
                # print(new_labels)
                # print([idx_to_label.get(key) for key in labels])
                # print([idx_to_label.get(key) for key in labels])
                # print(mask)
                # print(pos)
                # print(word_segment_ids)
                # print(tokens)
                # tokens = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                # print(tokens)
                # exit()
                assert len(tokens) == len(new_labels), AssertionError("The length of token and label is not matching")

                dataset.append([tokens, new_labels, mask, pos, vm, tag, word_segment_ids])

                # Enable dry rune
                if args.dry_run:
                    count += 1
                    if count == 100:
                        break

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, final=False):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([sample[3] for sample in dataset])
        vm_ids = torch.BoolTensor([sample[4] for sample in dataset])
        tag_ids = torch.LongTensor([sample[5] for sample in dataset])
        segment_ids = torch.LongTensor([sample[6] for sample in dataset])

        instances_num = input_ids.size(0)
        batch_size = args.batch_size

        if is_test:
            logger.info(f"Batch size:{batch_size}")
            print(f"The number of test instances:{instances_num}")

        true_labels_all = []
        predicted_labels_all = []
        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)
        model.eval()

        for i, (
                input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch,
                tag_ids_batch, segment_ids_batch) in enumerate(
            batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, segment_ids)):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)
            segment_ids_batch = segment_ids_batch.long().to(device)

            pred = model(input_ids_batch, segment_ids_batch, mask_ids_batch, label_ids_batch, pos_ids_batch,
                         vm_ids_batch, use_kg=args.use_kg)

            for pred_sample, gold_sample, mask in zip(pred, label_ids_batch, mask_ids_batch):

                pred_labels = [idx_to_label.get(key) for key in pred_sample.tolist()]
                gold_labels = [idx_to_label.get(key) for key in gold_sample.tolist()]

                num_labels = sum(mask)

                # Exclude the [CLS], and [SEP] tokens
                pred_labels = pred_labels[1:num_labels-1]
                true_labels = gold_labels[1:num_labels-1]

                pred_labels = [p.replace('_NOKG', '') for p in pred_labels]
                true_labels = [t.replace('_NOKG', '') for t in true_labels]

                true_labels, pred_labels = filter_kg_labels(true_labels, pred_labels)

                pred_labels = [p.replace('_', '-') for p in pred_labels]
                true_labels = [t.replace('_', '-') for t in true_labels]

                biluo_tags_predicted = get_bio(pred_labels)
                biluo_tags_true = get_bio(true_labels)

                if len(biluo_tags_predicted) != len(biluo_tags_true):
                    logger.error('The length of the predicted labels is not same as that of true labels..')
                    exit()

                predicted_labels_all.append(biluo_tags_predicted)
                true_labels_all.append(biluo_tags_true)

        if final:
            with open(f'{args.output_file_prefix}_predictions.txt', 'a') as p, \
                    open(f'{args.output_file_prefix}_gold.txt', 'a') as g:
                p.write('\n'.join([' '.join(l) for l in predicted_labels_all]))
                g.write('\n'.join([' '.join(l) for l in true_labels_all]))

        return dict(
            f1=seqeval.metrics.f1_score(true_labels_all, predicted_labels_all),
            precision=seqeval.metrics.precision_score(true_labels_all, predicted_labels_all),
            recall=seqeval.metrics.recall_score(true_labels_all, predicted_labels_all),
            f1_span=f1_score_span(true_labels_all, predicted_labels_all),
            precision_span=precision_score_span(true_labels_all, predicted_labels_all),
            recall_span=recall_score_span(true_labels_all, predicted_labels_all),
        )

    # Training phase.
    logger.info("Start training.")
    instances = read_dataset(args.train_path)

    input_ids = torch.LongTensor([ins[0] for ins in instances])
    label_ids = torch.LongTensor([ins[1] for ins in instances])
    mask_ids = torch.LongTensor([ins[2] for ins in instances])
    pos_ids = torch.LongTensor([ins[3] for ins in instances])
    vm_ids = torch.BoolTensor([ins[4] for ins in instances])
    tag_ids = torch.LongTensor([ins[5] for ins in instances])
    segment_ids = torch.LongTensor([ins[6] for ins in instances])

    instances_num = input_ids.size(0)
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    train_batcher = Batcher(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, segment_ids)

    logger.info(f"Batch size:{batch_size}")
    logger.info(f"The number of training instances:{instances_num}")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    best_f1 = 0.0

    # Dry evaluate
    # evaluate(args, True)

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (
                input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch,
                tag_ids_batch, segment_ids_batch) in enumerate(train_batcher):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)
            segment_ids_batch = segment_ids_batch.long().to(device)

            loss = model.score(input_ids_batch,
                               segment_ids_batch,
                               mask_ids_batch,
                               label_ids_batch,
                               pos_ids_batch,
                               vm_ids_batch,
                               use_kg=args.use_kg)

            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            optimizer.step()

        # Evaluation phase.
        logger.info("Start evaluate on dev dataset.")
        results = evaluate(args, False)
        logger.info(results)

        logger.info("Start evaluation on test dataset.")
        results_test = evaluate(args, True)
        logger.info(results_test)

        if results['f1'] > best_f1:
            best_f1 = results['f1']
            save_model(model, args.output_model_path)
            save_encoder(args, encoder, suffix=args.suffix_file_encoder)
        else:
            continue

    # Evaluation phase.
    logger.info("Final evaluation on test dataset.")
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    results_final = evaluate(args, True, final=True)
    logger.info(results_final)


if __name__ == "__main__":
    main()
