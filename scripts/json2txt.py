
import re

import jsonlines
import json

data_file_neg = f'/root/autodl-tmp/FT/air/air/test_data/new_generates/imdb_negative_64_ck7_JS.txt'
data_file_pos = f'/root/autodl-tmp/FT/air/air/test_data/new_generates/imdb_positive_64_ck7_JS.txt'



 
 定义JSONL文件和TXT文件的路径
jsonl_file_path = "/root/autodl-tmp/FT/air/air/test_data/new_generates/Air_sentiment_140.0_length_64_ck7_JS.jsonl"
pos_txt_file_path = data_file_pos
neg_txt_file_path = data_file_neg
 
 打开TXT文件进行写入
with open(pos_txt_file_path, 'w', encoding="utf-8-sig") as pos_file, open(neg_txt_file_path, 'w', encoding="utf-8-sig") as neg_file, open(jsonl_file_path, 'r', encoding="utf-8-sig") as jsonl_file:
    for line in jsonl_file:
         解析JSON数据
        data = json.loads(line)
        txt = data["text"].replace("\n","")
        if data["sentiment"] == "Positive":
            pos_file.write(txt + "\n")
        else:
            neg_file.write(txt + "\n")


