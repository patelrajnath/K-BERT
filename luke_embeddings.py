import torch

from luke import ModelArchive, LukeModel

model_archive = ModelArchive.load('D:\\Downloads\\luke_base_500k.tar.gz')
tokenizer = model_archive.tokenizer
target_tokens = "Hello Luke Model".split()

word_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + target_tokens + [tokenizer.sep_token])
word_attention_mask = [1] * (len(target_tokens) + 2)
word_segment_ids = [0] * (len(target_tokens) + 2)

word_ids = torch.LongTensor(word_ids).unsqueeze(0)
word_attention_mask = torch.LongTensor(word_attention_mask).unsqueeze(0)
word_segment_ids = torch.LongTensor(word_segment_ids).unsqueeze(0)

print(word_ids, word_attention_mask, word_segment_ids)

model = LukeModel(model_archive.config)

encoding = model(word_ids=word_ids,
                 word_attention_mask=word_attention_mask,
                 word_segment_ids=word_segment_ids)

print(encoding[0])
print(encoding[1])
print(encoding[0].size())
print(encoding[1].size())
# print(encoding[2].size())

