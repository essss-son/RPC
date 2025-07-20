
import re

import jsonlines
import json
import argparse
import os


def main(args):
    data_file_neg = f'/root/autodl-tmp/FT/air/test_data/rand/air_neg_64.txt'
    data_file_pos = f'/root/autodl-tmp/FT/air/test_data/rand/air_pos_64.txt'
    data_file_neg = os.path.join(args.gen_dir_name, args.attr0_txt)
    data_file_pos = os.path.join(args.gen_dir_name, args.attr1_txt)
    jsonl_file_path = os.path.join(args.gen_dir_name, args.json_file_name)



 
     定义JSONL文件和TXT文件的路径
    jsonl_file_path = "/root/autodl-tmp/FT/air/test_data/attr2/Air_sentiment_140.0_length_64.jsonl"
    pos_txt_file_path = data_file_pos
    neg_txt_file_path = data_file_neg
     
     打开TXT文件进行写入
    with open(pos_txt_file_path, 'w', encoding="utf-8-sig") as pos_file, open(neg_txt_file_path, 'w', encoding="utf-8-sig") as neg_file, open(jsonl_file_path, 'r', encoding="utf-8-sig") as jsonl_file:
        for line in jsonl_file:
             解析JSON数据
            data = json.loads(line)
            txt = data["text"].replace("\n"," ")
            if data["sentiment"] == "Positive":
                pos_file.write(txt + "\n")
            else:
                neg_file.write(txt + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir_name", default="/root/autodl-tmp/FT/air/test_data/attr2/", type=str)
    parser.add_argument("--attr0_txt", default="gpt2_medium_imdb_attr0.txt", type=str)
    parser.add_argument("--attr1_txt", default="gpt2_medium_imdb_attr1.txt", type=str)
    parser.add_argument("--json_file_name", default="gpt2_medium_topic.jsonl", type=str)
    args = parser.parse_args()
    main(args)