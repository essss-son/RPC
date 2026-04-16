import os

from model_sent import RobertaForPreTraining
from transformers import RobertaTokenizer
# from train import tokendataset, padding_fuse_fn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# from evaluate import load
import argparse
import torch
import json
import pdb

# DEVICE = torch.device("cuda:1")
MAXLEN = 512
BATCH_SIZE = 4

class tokendataset(Dataset):
    def __init__(self, dataset_path):
        file_row_pos = 0
        file_row_neg = 0
        for item in dataset_path:
            sentiment = item['sentiment']
            if sentiment == 0:
                file_row_neg += 1
            elif sentiment == 1:
                file_row_pos += 1
            else:
                raise Exception("Sentiment Type Error!")
            
        self.file_row = len(dataset_path)
        self.file_row_pos = file_row_pos
        self.file_row_neg = file_row_neg
        self.dataset = dataset_path

    def __len__(self):
        return self.file_row

    def __getitem__(self, idx):
        return self.dataset[idx]

def padding_fuse_fn(data_list):
    input_ids = []
    attention_masks = []
    sentiment = []
    text_length = []
    ppl = []
    for item in data_list:
        text_length.append(len(item["text"]))
        sentiment.append([item["sentiment"]])
    max_text_len = max(text_length)

    for i, item in enumerate(data_list):
        text_pad_len = max_text_len - text_length[i]

        attention_mask = [1] * text_length[i] + [0] * text_pad_len
        text = item["text"] + [0] * text_pad_len

        input_ids.append(text)
        attention_masks.append(attention_mask)
    
    batch = {}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_masks
    batch["sentiment"] = sentiment

    return batch

def tokenized(dataset_path=None, tokenizer=None):

    output_data = list()

    f = open(dataset_path, 'r')
    for line in f:
        data = {}
        dic = json.loads(line)
        if dic['sentiment'] == 'Negative':
            data['sentiment'] = 0
        elif dic['sentiment'] == 'Positive':
            data['sentiment'] = 1

        if 'text' in dic:
            data['text'] = tokenizer.encode(dic['text'], max_length=MAXLEN, truncation=True)
        elif 'review' in dic:
            data['text'] = tokenizer.encode(dic['review'], max_length=MAXLEN, truncation=True)

        output_data.append(data)
    
    f.close()

    return output_data

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--dataset_path", default="/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_0.5_384.json", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--device_num", default=0, type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda")

    # tokenized the data in dataset_path

    args.model_name_or_path = "/home/anke/DXZ/RPC/Air-Decoding-main/models/best_sentiment_classifier"
    # args.dataset_path = "/home/anke/DXZ/RPC/Air-Decoding-main/test_data/texts.jsonl"
    # args.batch_size = 4

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    output_data = tokenized(dataset_path=args.dataset_path, tokenizer=tokenizer)
    args.output_data = output_data
    # pdb.set_trace()

    test_dataset = tokendataset(args.output_data)
    file_row = test_dataset.file_row
    file_row_pos = test_dataset.file_row_pos
    file_row_neg = test_dataset.file_row_neg
    test_sampler = torch.utils.data.RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=test_sampler)

    model = RobertaForPreTraining.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    tp_all, tp0_all, tp1_all, fp0_all, fp1_all = 0, 0, 0, 0, 0
    tr_loss = 0.0
    logs = {}

    for step, batch in enumerate(test_dataloader):
        input_ids, attention_mask, sentiment = batch['input_ids'], batch['attention_mask'], batch['sentiment']

        input_ids = torch.tensor(input_ids).to(args.device)
        attention_mask = torch.tensor(attention_mask).to(args.device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        sentiment = torch.tensor(sentiment).to(args.device)

        loss, tp, tp0, tp1, fp0, fp1 = model(input_ids=input_ids, attention_mask=attention_mask, sentiment=sentiment)
        tp_all += tp
        tp0_all += tp0
        tp1_all += tp1
        fp0_all += fp0
        fp1_all += fp1

        tr_loss += loss.item()

    acc = tp_all / (tp_all + fp0_all + fp1_all)
    acc0 = tp0_all / (tp0_all + fp0_all)
    acc1 = tp1_all / (tp1_all + fp1_all)
    logs['acc'] = float('{:.4f}'.format(acc))
    logs['acc0'] = float('{:.4f}'.format(acc0))
    logs['acc1'] = float('{:.4f}'.format(acc1))
    logs['total_loss'] = tr_loss

    length = int(args.dataset_path[-8:-5])
    insert_prob = float(args.dataset_path[-12:-9])
    write_path = "./result.json"
    # print(os.getcwd())
    with open(write_path, "a") as f:
        obj ={
            "file":f"insert_prob_{insert_prob}_length_{length}.json",
            "acc":float('{:.4f}'.format(acc))
        }
        json.dump(obj, f)
        f.write("\n")


    print(logs)
    
if __name__ == "__main__":
    main()