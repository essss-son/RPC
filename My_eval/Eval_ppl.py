import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
class mydataset(Dataset):
    def __init__(self, text_list):
        self.text_list = text_list

    def __len__(self):
        return len(self.text_list)
    def __getitem__(self, item):
        return self.text_list[item]


def main(args):
    text = []
    args.tokenizer.pad_token = args.tokenizer.eos_token
    file_path = os.path.join('./output', args.eval_file_path)

    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            text.append(obj['text'])
    ppl_list = []
    dataset = mydataset(text)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    for batch in tqdm(dataloader,total=len(dataset) // args.batch_size):
        encoded = args.tokenizer(batch, padding=True, return_tensors='pt').to(args.device)
        logits = args.model(**encoded).logits[:,:-1,:]
        mask = encoded['attention_mask'][:,1:]
        mask_num = mask.sum(dim=-1,keepdim=True)

        probs = logits.softmax(dim=-1)
        target = encoded['input_ids'][:,1:]
        select_probs = probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        # size batch_size, seq_len
        ppl = torch.exp(-(torch.log(select_probs) * mask).sum(dim=1,keepdim=True) / mask_num).squeeze(-1).mean().item()
        # size batch_size, 1
        ppl_list.append(ppl)

    # print(f"ave ppl:{np.array(ppl_list).mean()}")
    ave_ppl = np.array(ppl_list).mean()
    eval_length = int(args.eval_file_path[-8:-5])
    eval_insert_prob = float(args.eval_file_path[-12:-9])
    result_path = f"sentiment_eval_result.json"
    with open(result_path, "a", encoding="utf-8") as f:
        obj = {
            "file": "insert_prob_{}_length_{}".format(eval_insert_prob, eval_length),
            "ppl": ave_ppl,
        }
        json.dump(obj, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None,help='gpt2-medium')
    parser.add_argument('--eval_file_path', type=str, default="sentiment_output_0.5_128.json")
    parser.add_argument('--batch_size', type=int, default=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.device = device
    args.model_path = "/home/anke/DXZ/models/gpt2-medium"
    # args.eval_file_path = "/home/anke/DXZ/RPC/rpc-new/output/sentiment_output.json"

    args.tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    args.model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

    main(args)
