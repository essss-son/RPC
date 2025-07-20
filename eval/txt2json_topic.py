import json
import re
import os
import jsonlines
import argparse

            
def main(args):
    data_file_0 = os.path.join(args.gen_dir_name, args.World_txt)
    data_file_1 = os.path.join(args.gen_dir_name, args.Sports_txt)
    data_file_2 = os.path.join(args.gen_dir_name, args.Business_txt)
    data_file_3 = os.path.join(args.gen_dir_name, args.Science_txt)
    file_jsonl_path = "/root/autodl-tmp/attr2/test_gen/decoding/gpt2_medium_imdb.jsonl"
    file_jsonl_path = os.path.join(args.gen_dir_name, args.json_file_name)
    
    with jsonlines.open(file_jsonl_path, mode="w") as file_jsonl:
        with open(data_file_0, 'r', encoding="utf-8-sig") as file:
            for line in file:
                item = {
                    "text": line.strip(),
                    "topic": "World"
                }
                file_jsonl.write(item)


        with open(data_file_1, 'r', encoding="utf-8-sig") as file:
            for line in file:
                item = {
                    "text": line.strip(),
                    "topic": "Sports"
                }
                file_jsonl.write(item)

        with open(data_file_2, 'r', encoding="utf-8-sig") as file:
            for line in file:
                item = {
                    "text": line.strip(),
                    "topic": "Business"
                }
                file_jsonl.write(item)


        with open(data_file_3, 'r', encoding="utf-8-sig") as file:
            for line in file:
                item = {
                    "text": line.strip(),
                    "topic": "Science"
                }
                file_jsonl.write(item)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir_name", default="/root/autodl-tmp/attr2/test_gen/decoding", type=str)
    parser.add_argument("--World_txt", default=None, type=str)
    parser.add_argument("--Sports_txt", default=None, type=str)
    parser.add_argument("--Business_txt", default=None, type=str)
    parser.add_argument("--Science_txt", default=None, type=str)
    parser.add_argument("--json_file_name", default="gpt2_medium_topic.jsonl", type=str)
    args = parser.parse_args()
    main(args)