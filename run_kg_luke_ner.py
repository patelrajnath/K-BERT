# -*- encoding:utf -*-
"""
  This script provides an K-BERT example for NER.
"""
import argparse
import torch
import torch.nn as nn

from brain import config
from brain.knowgraph_english import KnowledgeGraph
from luke import ModelArchive, LukeModel
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import BertAdam
from uer.utils.constants import *
from uer.utils.seed import set_seed
from uer.model_saver import save_model


class LukeTagger(nn.Module):
    def __init__(self, args, encoder):
        super(LukeTagger, self).__init__()
        self.encoder = encoder
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,
                word_ids,
                word_segment_ids,
                word_attention_mask,
                label,
                pos=None,
                vm=None,
                use_kg=True
                ):
        if not use_kg:
            vm = None

        # Encoder.
        word_sequence_output, pooled_output = self.encoder(word_ids, word_segment_ids=word_segment_ids,
                                                           word_attention_mask=word_attention_mask,
                                                           position_ids=pos, vm=vm)
        # Target.
        output = self.output_layer(word_sequence_output)

        output = output.contiguous().view(-1, self.labels_num)
        output = self.softmax(output)

        label = label.contiguous().view(-1, 1)
        label_mask = (label > 0).float().to(torch.device(label.device))
        one_hot = torch.zeros(label_mask.size(0), self.labels_num). \
            to(torch.device(label.device)). \
            scatter_(1, label, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        predict = output.argmax(dim=-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )

        return loss, correct, predict, label


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/tagger_model.bin", type=str,
                        help="Path of the output model.")
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

    # Model options.
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=256, type=int,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")

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
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--use_kg", action='store_true', help="Enable the use of KG.")
    parser.add_argument("--debug", action='store_true', help="Enable debug.")

    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

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
        spo_files = []
    else:
        spo_files = args.kg_name

    # Load Luke model.
    model_archive = ModelArchive.load(args.pretrained_model_path)
    tokenizer = model_archive.tokenizer
    encoder = LukeModel(model_archive.config)

    # Build sequence labeling model.
    model = LukeTagger(args, encoder)
    kg = KnowledgeGraph(vocab_file=spo_files, tokenizer=tokenizer)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        with open(path, mode="r", encoding="utf8") as f:
            f.readline()
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                labels, tokens, cls = line.strip().split("\t")

                tokens, pos, vm, tag = \
                    kg.add_knowledge_with_vm([tokens], [labels],
                                             use_kg=args.use_kg,
                                             max_length=args.seq_length
                                             )
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]

                # tokens = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                non_pad_tokens = [tok for tok in tokens if tok != tokenizer.pad_token]
                num_tokens = len(non_pad_tokens)
                num_pad = len(tokens) - num_tokens

                labels = [labels_map[config.CLS_TOKEN]] + \
                         [labels_map[l] for l in labels.split(" ")] + \
                         [labels_map[config.SEP_TOKEN]]

                # print(tokens)
                # print(labels)
                # print(tag)

                mask = [1] * (num_tokens) + [0] * num_pad
                word_segment_ids = [0] * (len(tokens))
                new_labels = []
                j = 0

                # print(len(tokens))
                # print(len(tag))
                # print(tokenizer.pad_token_id)

                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != tokenizer.pad_token:
                        new_labels.append(labels[j])
                        j += 1
                    elif tag[i] == 1 and tokens[i] != tokenizer.pad_token:  # 是添加的实体
                        new_labels.append(labels_map['[ENT]'])
                    elif tag[i] == 2:
                        new_labels.append(labels_map['[X]'])
                    else:
                        new_labels.append(labels_map[PAD_TOKEN])
                # print(new_labels)

                # tokens = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                dataset.append([tokens, new_labels, mask, pos, vm, tag, word_segment_ids])

        return dataset

    # Evaluation function.
    def evaluate(args, is_test):
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
            print("Batch size: ", batch_size)
            print("The number of test instances:", instances_num)

        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0

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

            loss, _, pred, gold = model(input_ids_batch,
                                        segment_ids_batch,
                                        mask_ids_batch,
                                        label_ids_batch,
                                        pos_ids_batch,
                                        vm_ids_batch,
                                        use_kg=args.use_kg
                                        )
            with open('predictions.txt', 'a+') as p:
                predicted_labels = [idx_to_label.get(key) for key in pred.tolist()]
                p.write(' '.join(predicted_labels) + '\n')

            with open('gold.txt', 'a+') as p:
                gold_labels = [idx_to_label.get(key) for key in gold.tolist()]
                p.write(' '.join(gold_labels) + '\n')

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    gold_entities_num += 1

            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:
                    pred_entities_num += 1

            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    start = j
                    for k in range(j + 1, gold.size()[0]):

                        if gold[k].item() == labels_map['[ENT]'] or gold[k].item() == labels_map['[X]']:
                            continue

                        if gold[k].item() == labels_map["[PAD]"] or gold[k].item() == labels_map["O"] or gold[
                            k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = gold.size()[0] - 1
                    gold_entities_pos.append((start, end))

            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"] and gold[j].item() != \
                        labels_map["[ENT]"] and gold[j].item() != labels_map["[X]"]:
                    start = j
                    for k in range(j + 1, pred.size()[0]):

                        if gold[k].item() == labels_map['[ENT]'] or gold[k].item() == labels_map['[X]']:
                            continue

                        if pred[k].item() == labels_map["[PAD]"] or pred[k].item() == labels_map["O"] or pred[
                            k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = pred.size()[0] - 1
                    pred_entities_pos.append((start, end))

            for entity in pred_entities_pos:
                if entity not in gold_entities_pos:
                    continue
                else:
                    correct += 1

        print("Report precision, recall, and f1:")
        p = correct / pred_entities_num + 1  # + 1 is avoid divide by zero error
        r = correct / gold_entities_num
        f1 = 2 * p * r / (p + r)
        print("{:.3f}, {:.3f}, {:.3f}".format(p, r, f1))

        return f1

    # Training phase.
    print("Start training.")
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

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    f1 = 0.0
    best_f1 = 0.0

    # Dry evaluate
    # evaluate(args, True)

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (
                input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch,
                tag_ids_batch, segment_ids_batch) in enumerate(
            batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids, segment_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)
            segment_ids_batch = segment_ids_batch.long().to(device)

            loss, _, _, _ = model(input_ids_batch,
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
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                total_loss = 0.

            loss.backward()
            optimizer.step()

        # Evaluation phase.
        print("Start evaluate on dev dataset.")
        f1 = evaluate(args, False)
        print("Start evaluation on test dataset.")
        evaluate(args, True)

        if f1 > best_f1:
            best_f1 = f1
            save_model(model, args.output_model_path)
        else:
            continue

    # Evaluation phase.
    print("Final evaluation on test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))

    evaluate(args, True)


if __name__ == "__main__":
    main()
