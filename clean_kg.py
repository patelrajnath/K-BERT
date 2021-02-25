import json

vocab_file = 'D:\\Downloads\\ent_vocab_custom'
lookup_table = {}
count = 0
with open(vocab_file, "r", encoding='utf-8') as f:
    entities_json = [json.loads(line) for line in f]
    for item in entities_json:
        for title, language in item["entities"]:
            value = item["info_box"]
            for key in value:
                try:
                    lookup_table[key] += 1
                except:
                    lookup_table[key] = 1
        if count == 100:
            break
        count += 1

lookup_table_sorted = {k: v for k, v in sorted(lookup_table.items(), key=lambda item: item[1], reverse=True)}
print(lookup_table_sorted)
count = 0
with open(vocab_file, "r", encoding='utf-8') as f:
    entities_json = [json.loads(line) for line in f]
    for item in entities_json:
        temp = {}
        for title, language in item["entities"]:
            print(title)
            value = item["info_box"]
            for key in value:
                temp[key] = lookup_table.get(key)
            sorted_value = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1], reverse=True)}
            print(sorted_value)
            info_box = {}
            for key in sorted_value:
                info_box[key] = value[key]
            print(info_box)
        if count == 100:
            break
        count += 1
