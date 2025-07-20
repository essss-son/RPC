
import re

import jsonlines
import json
import argparse
import os


def main(args):
    data_file_world = os.path.join(args.gen_dir_name, args.World_txt)
    data_file_sports = os.path.join(args.gen_dir_name, args.Sports_txt)
    data_file_business = os.path.join(args.gen_dir_name, args.Business_txt)
    data_file_science = os.path.join(args.gen_dir_name, args.Science_txt)
    file_jsonl_path = "/root/autodl-tmp/attr2/test_gen/decoding/gpt2_medium_imdb.jsonl"
    jsonl_file_path = os.path.join(args.gen_dir_name, args.json_file_name)



 
     定义JSONL文件和TXT文件的路径

     
     打开TXT文件进行写入
    with open(data_file_world, 'w', encoding="utf-8-sig") as world_file, open(data_file_sports, 'w', encoding="utf-8-sig") as sports_file, open(data_file_business, 'w', encoding="utf-8-sig") as business_file, open(data_file_science, 'w', encoding="utf-8-sig") as science_file, open(jsonl_file_path, 'r', encoding="utf-8-sig") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            txt = data["text"].replace("\n"," ")
            if data["topic"] == "World":
                world_file.write(txt + "\n")
            elif data["topic"] == "Sports":
                sports_file.write(txt + "\n")
            elif data["topic"] == "Business":
                business_file.write(txt + "\n")
            elif data["topic"] == "Science":
                science_file.write(txt + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir_name", default="/root/autodl-tmp/FT/air/test_data/attr2/", type=str)
    parser.add_argument("--World_txt", default=None, type=str)
    parser.add_argument("--Sports_txt", default=None, type=str)
    parser.add_argument("--Business_txt", default=None, type=str)
    parser.add_argument("--Science_txt", default=None, type=str)
    parser.add_argument("--json_file_name", default="gpt2_medium_topic.jsonl", type=str)
    args = parser.parse_args()
    main(args)