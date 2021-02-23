# coding: utf-8
"""
KnowledgeGraph
"""
import json
import brain.config as config
import numpy as np
from datautils import biluo_from_predictions, Doc
from datautils.biluo_from_predictions import get_biluo
from datautils.iob_utils import offset_from_biluo


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, vocab_file, tokenizer, predicate=False):
        self.predicate = predicate
        self.vocab_file = vocab_file
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.tokenizer = tokenizer

    def _create_lookup_table(self):
        lookup_table = {}
        with open(self.vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]
        for item in entities_json:
            for title, language in item["entities"]:
                value = item["info_box"]
                if value:
                    lookup_table[title.lower()] = value
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, label_batch,
                              max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        text_ = sent_batch[0]
        label_ = label_batch[0]

        tag_labels_true = label_.strip().replace('_', '-').split()
        biluo_tags_true = get_biluo(tag_labels_true)

        doc = Doc(text_)
        offset_true_labels = offset_from_biluo(doc, biluo_tags_true)

        chunk_start = 0
        chunks = []

        # Convert text into chunks
        for start, end, _ in offset_true_labels:
            chunk_text = text_[chunk_start: start].strip()
            chunk_entity = text_[start: end].strip()
            chunk_start = end

            if chunk_text:
                chunks.append(chunk_text)

            if chunk_entity:
                chunks.append(chunk_entity)

        # Append the last chunk if not empty
        last_chunk = text_[chunk_start:].strip()
        if last_chunk:
            chunks.append(last_chunk)
        chunks = [chunks]

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in chunks:
            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            print(split_sent)
            for token_original in split_sent:
                entities = list(self.lookup_table.get(token_original.lower(), []))[:max_entities]
                entities = [ent.replace('_', ' ') for ent in entities]

                print(entities, token_original)

                # Tokenize the data
                cur_tokens = []
                for tok in token_original.split():
                    cur_tokens.extend(self.tokenizer.tokenize(tok))

                entities = [self.tokenizer.tokenize(ent) for ent in entities]

                sent_tree.append((token_original, cur_tokens, entities))

                if token_original in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(cur_tokens)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(cur_tokens)+1)]
                print(token_abs_idx)
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                print(f'token_abs_idx:{token_abs_idx}')
                print(f'token_pos_idx:{token_pos_idx}')
                print(f'entities_abs_idx:{entities_abs_idx}')
                print(f'entities_pos_idx:{entities_pos_idx}')

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            print(abs_idx_tree)
            print(pos_idx_tree)
            print(sent_tree)

            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                token_original = sent_tree[i][0]
                word = sent_tree[i][1]

                for tok in token_original.split():
                    cur_toks = self.tokenizer.tokenize(tok)
                    num_subwords = len(cur_toks)
                    if tok in self.special_tags:
                        seg += [0]
                    else:
                        seg += [0]
                        # Add extra tags for the added subtokens
                        if num_subwords > 1:
                            seg += [2] * (num_subwords - 1)

                if token_original in self.special_tags:
                    know_sent += word
                else:
                    add_word = word
                    know_sent += add_word

                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][2])):
                    add_word = sent_tree[i][2][j]
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)
            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [self.tokenizer.pad_token] * pad_num
                seg += [3] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch
