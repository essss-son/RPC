import json
import re
import os

import jsonlines

dir_name = '/root/autodl-tmp/FT/attr/gen/sent_20prefix/512-mid-dim-256data-10epochs-256length/'

neg_txt = 'gpt2_medium_imdb_attr0.txt'
pos_txt = 'gpt2_medium_imdb_attr1.txt'


data_file_neg = os.path.join(dir_name, neg_txt)
data_file_pos = os.path.join(dir_name, pos_txt)


file_jsonl = "gpt2_medium_imdb_256.jsonl"
file_jsonl_path = os.path.join(dir_name, file_jsonl)

with open(data_file_neg, 'r', encoding="utf-8-sig") as file:
    for line in file:
        item = {
            "text": line.strip(),
            "sentiment": "Negative"
        }
        with jsonlines.open(file_jsonl_path, mode="a") as file_jsonl:
            file_jsonl.write(item)


with open(data_file_pos, 'r', encoding="utf-8-sig") as file:
    for line in file:
        item = {
            "text": line.strip(),
            "sentiment": "Positive"
        }
        with jsonlines.open(file_jsonl_path, mode="a") as file_jsonl:
            file_jsonl.write(item)

