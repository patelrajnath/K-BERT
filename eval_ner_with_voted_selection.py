# Load Luke model.
import unicodedata
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


# file_in = '../data/combined_3/test_combined_3.csv'
# file_out = '../data/combined_3/test_combined_3_text_tokenized.txt'
# file_in = '../data/nlu/nlu_test_bio.csv'
# file_out = '../data/nlu/nlu_test_bio_tokenized.txt'
file_in = '../data/nlu/nlu_test.csv'
file_out = '../data/nlu/nlu_test_tokenized.txt'

with open(file_in, mode="r", encoding="utf8") as f, \
        open(file_out, mode="w", encoding="utf8") as fout:
    f.readline()
    tokens, labels = [], []
    text_tokenized = []
    for line_id, line in enumerate(f):
        tokens_subword = []

        fields = line.strip().split("\t")
        if len(fields) == 2:
            labels, tokens = fields
        elif len(fields) == 3:
            labels, tokens, cls = fields
        else:
            print(f'The data is not in accepted format at line no:{line_id}.. Ignored')
            continue
        for token in tokens.split():
            sub_tokens = tokenize_word(token)
            tokens_subword.extend(sub_tokens)
        tokens_subword = ['<s>'] + tokens_subword + ['</s>']
        fout.write(' '.join(tokens_subword) + '\n')
