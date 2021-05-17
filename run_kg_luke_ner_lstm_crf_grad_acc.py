# -*- encoding:utf -*-
"""
  This script provides an K-BERT example for NER.
"""
import argparse
import contextlib
import logging
import os
import sys

import numpy
import seqeval
import torch
import torch.nn as nn

from collections import Counter

from seqeval.metrics import f1_score
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW

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


def create_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        eps=args.adam_eps,
        betas=(args.adam_b1, args.adam_b2),
        correct_bias=args.adam_correct_bias,
    )


def create_scheduler(args, optimizer):
    warmup_steps = int(args.num_train_steps * args.warmup_proportion)
    if args.lr_schedule == "warmup_linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, args.num_train_steps)
    if args.lr_schedule == "warmup_constant":
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)
    raise RuntimeError("Unsupported scheduler: " + args.lr_schedule)


class Batcher(object):
    def __init__(self, batch_size, instances, token_pad, label_pad, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.token_pad = token_pad
        self.label_pad = label_pad
        self.data = numpy.asarray(instances, dtype=object)
        self.num_samples = len(instances)
        self._indices = numpy.arange(self.num_samples)
        self.rnd = numpy.random.RandomState(0)
        self.indices = torch.randperm(self.num_samples)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.num_samples:
            if self.shuffle:
                self.rnd.shuffle(self._indices)
            self.ptr = 0
            raise StopIteration
        else:
            batch_indices = self.indices[self.ptr: self.ptr + self.batch_size]
            self.ptr += self.batch_size
            # input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, \
            # tag_ids_batch, segment_ids_batch = self.data[batch_indices]
            batch = self.data[batch_indices]

            # compute length of longest sentence in batch
            max_length = max([len(s[0]) for s in batch])

            labels = self.label_pad * numpy.ones((len(batch), max_length))

            # Dynamic batching
            for index, example in enumerate(batch):
                input_ids, _, _, _, vm_ids, _, _ = example
                current_length = len(input_ids)

                pad_num = max_length - current_length
                batch[index][0] += [self.token_pad] * pad_num
                labels[index][:current_length] = batch[index][1]

                batch[index][1] = labels[index]
                batch[index][2] = [1] * current_length + [0] * pad_num

                batch[index][3] += [max_length - 1] * pad_num
                batch[index][4] = numpy.pad(vm_ids, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
                batch[index][6] = [0] * max_length

            batch_input_ids = torch.LongTensor([sample[0] for sample in batch])
            batch_label_ids = torch.LongTensor([sample[1] for sample in batch])
            batch_mask_ids = torch.LongTensor([sample[2] for sample in batch])
            batch_pos_ids = torch.LongTensor([sample[3] for sample in batch])
            batch_vm_ids = torch.BoolTensor([sample[4] for sample in batch])
            batch_segment_ids = torch.LongTensor([sample[6] for sample in batch])

            del labels
            del batch
            del vm_ids

            return batch_input_ids, batch_label_ids, batch_mask_ids, \
                   batch_pos_ids, batch_vm_ids, batch_segment_ids


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
        bs, seq_len = word_ids.shape
        logits = self.get_logits(tensor)
        logits = logits.contiguous().view(-1, self.labels_num)
        outputs = F.log_softmax(logits, dim=-1)
        predict = outputs.argmax(dim=-1).view(-1, seq_len)
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
        self.args = args
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
        self.args = args
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
        self.args = args
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
    parser.add_argument("--pretrained_model_path", default=None, type=str, required=True,
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
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--lr_schedule", default="warmup_linear", type=str,
                        choices=["warmup_linear", "warmup_constant"])
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--max_grad_norm", default=0.0, type=float)
    parser.add_argument("--adam_b1", default=0.9, type=float)
    parser.add_argument("--adam_b2", default=0.98, type=float)
    parser.add_argument("--adam_eps", default=1e-6, type=float)
    parser.add_argument("--adam_correct_bias", action='store_true')
    parser.add_argument("--warmup_proportion", default=0.06, type=float)
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of steps to accumulate the gradient.")
    parser.add_argument("--report_steps", type=int, default=2,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=35,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--use_kg", action='store_true', help="Enable the use of KG.")
    parser.add_argument("--padding", action='store_true', help="Enable padding.")
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
    for file in (args.train_path, args.dev_path, args.test_path):
        with open(file, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                labels = line.strip().split("\t")[0].split()
                for l in labels:
                    if l not in labels_map:
                        if l.startswith("B") or l.startswith("S"):
                            begin_ids.append(len(labels_map))
                            # check if I-TAG exists
                            infix = l[1]
                            tag = l[2:]
                            inner_tag = f'I{infix}{tag}'
                            if inner_tag not in labels_map:
                                labels_map[inner_tag] = len(labels_map)

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
                                             reverse_order=args.reverse_order,
                                             padding=args.padding)
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
                        if cur_type != 'O':
                            try:
                                joiner = cur_type[1]
                                prev_label = cur_type[2:]
                            except:
                                logger.info(f'The label:{cur_type} is converted to O')
                                prev_label = 'O'
                                j += 1
                                new_labels.append('O')
                                continue
                        else:
                            prev_label = cur_type

                        new_labels.append(cur_type)
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

        instances_num = len(dataset)
        batch_size = args.batch_size

        if is_test:
            logger.info(f"Batch size:{batch_size}")
            print(f"The number of test instances:{instances_num}")

        true_labels_all = []
        predicted_labels_all = []
        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)
        model.eval()

        test_batcher = Batcher(batch_size, dataset, shuffle=False,
                               token_pad=tokenizer.pad_token_id, label_pad=labels_map[PAD_TOKEN])

        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch,
                vm_ids_batch, segment_ids_batch) in enumerate(test_batcher):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
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

    instances_num = len(instances)
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1
    args.num_train_steps = train_steps

    train_batcher = Batcher(batch_size, instances, shuffle=True,
                            token_pad=tokenizer.pad_token_id, label_pad=labels_map[PAD_TOKEN])

    logger.info(f"Batch size:{batch_size}")
    logger.info(f"The number of training instances:{instances_num}")

    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer)
    total_loss = 0.
    best_f1 = 0.0

    # Dry evaluate
    # evaluate(args, True)

    def maybe_no_sync(step):
        if (
                hasattr(model, "no_sync")
                and (step + 1) % args.gradient_accumulation_steps != 0
        ):
            return model.no_sync()
        else:
            return contextlib.ExitStack()

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for step, (
                input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch,
                segment_ids_batch) in enumerate(train_batcher):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
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

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with maybe_no_sync(step):
                loss.backward()

            total_loss += loss.item()

            if (step + 1) % args.report_steps == 0:
                logger.info("Epoch id: {}, Training steps: {}, Avg loss: "
                            "{:.3f}".format(epoch, step + 1, total_loss / args.report_steps))
                total_loss = 0.

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

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
