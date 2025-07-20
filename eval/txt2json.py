import json
import re

import jsonlines
import argparse
import os

def main(args):
    data_file_neg = os.path.join(args.gen_dir_name, args.attr0_txt)
    data_file_pos = os.path.join(args.gen_dir_name, args.attr1_txt)
    file_jsonl_path = "/root/autodl-tmp/attr2/test_gen/decoding/gpt2_medium_imdb.jsonl"
    file_jsonl_path = os.path.join(args.gen_dir_name, args.json_file_name)
    
    with jsonlines.open(file_jsonl_path, mode="w") as file_jsonl:
        with open(data_file_neg, 'r', encoding="utf-8-sig") as file:
            for line in file:
                item = {
                    "text": line.strip(),
                    "sentiment": "Negative"
                }
                file_jsonl.write(item)


        with open(data_file_pos, 'r', encoding="utf-8-sig") as file:
            for line in file:
                item = {
                    "text": line.strip(),
                    "sentiment": "Positive"
                }
                file_jsonl.write(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir_name", default="/root/autodl-tmp/attr2/test_gen/decoding", type=str)
    parser.add_argument("--attr0_txt", default="gpt2_medium_imdb_attr0.txt", type=str)
    parser.add_argument("--attr1_txt", default="gpt2_medium_imdb_attr1.txt", type=str)
    parser.add_argument("--json_file_name", default="gpt2_medium_imdb.jsonl", type=str)
    args = parser.parse_args()
    main(args)