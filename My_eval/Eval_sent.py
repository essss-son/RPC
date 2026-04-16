import argparse
import json
import torch
import os
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification


import torch.nn.functional as F
def main(args):
    pos_texts = []
    neg_texts = []

    file_path = os.path.join("./output/", args.eval_file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj['sentiment'] == 'Positive':
                pos_texts.append(obj['text'])
            else:
                neg_texts.append(obj['text'])

    tokenizer = GPT2Tokenizer.from_pretrained(args.eval_model)
    model = GPT2ForSequenceClassification.from_pretrained(args.eval_model).to(args.device)
    model.eval()
    with torch.no_grad():
        num_pos_batches = len(pos_texts)//args.eval_batch_size
        pos_total_num = 0
        pos_num = 0
        for i in range(num_pos_batches):
            pos_batch = pos_texts[i*args.eval_batch_size:(i+1)*args.eval_batch_size]
            len_batch = len(pos_batch)
            pos_total_num += len_batch

            pos_encoded = tokenizer(pos_batch, return_tensors="pt", padding=True).to(args.device)
            logits = model(**pos_encoded).logits
            probs = logits.softmax(dim=-1)
            # size [batch_size, 2]
            pos_index = torch.argmax(probs, dim=-1).squeeze(-1)
            pos_num += pos_index.sum().item()

        pos_acc = round(pos_num / pos_total_num * 100, 2)

        num_neg_batches = len(neg_texts) // args.eval_batch_size
        neg_total_num = 0
        neg_num = 0
        for i in range(num_neg_batches):
            neg_batch = neg_texts[i * args.eval_batch_size:(i + 1) * args.eval_batch_size]
            len_batch = len(neg_batch)
            neg_total_num += len_batch

            neg_encoded = tokenizer(neg_batch, return_tensors="pt", padding=True).to(args.device)
            logits = model(**neg_encoded).logits
            probs = logits.softmax(dim=-1)
            # size [batch_size, 2]
            neg_index = 1 - torch.argmax(probs, dim=-1).squeeze(-1)
            neg_num += neg_index.sum().item()

        neg_acc = round(neg_num / neg_total_num * 100, 2)
        total_acc = round((pos_acc + neg_acc) / 2, 2)

    # with torch.no_grad():
    #     pos_encoded = tokenizer(pos_texts, return_tensors="pt", padding=True).to(args.device)
    #     logits = model(**pos_encoded).logits
    #     probs = logits.softmax(dim=-1)
    #     #size [batch_size, 2]
    #     pos_index = torch.argmax(probs, dim=-1).squeeze(-1)
    #     pos_num = pos_index.sum().item()
    #
    #     neg_encoded = tokenizer(neg_texts, return_tensors="pt", padding=True).to(args.device)
    #     logits = model(**neg_encoded).logits
    #     probs = logits.softmax(dim=-1)
    #     # size [batch_size, 2]
    #     neg_index = 1 - torch.argmax(probs, dim=-1).squeeze(-1)
    #     neg_num = neg_index.sum().item()
    #
    #     pos_acc = round(pos_num / len(pos_texts)*100,2)
    #     neg_acc = round(neg_num / len(neg_texts)*100,2)

    # print(f"ave acc:{total_acc}, pos acc:{pos_acc}, neg acc:{neg_acc}")
    eval_length = int(args.eval_file_path[-8:-5])
    eval_insert_prob = float(args.eval_file_path[-12:-9])
    result_path = f"sentiment_eval_result.json"
    with open(result_path, "a", encoding="utf-8") as f:

        obj = {
            "file":"insert_prob_{}_length_{}".format(eval_insert_prob,eval_length),
            "total_acc":total_acc,
            "pos_acc":pos_acc,
            "neg_acc":neg_acc,
        }
        json.dump(obj,f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model',type=str,default="/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment")
    parser.add_argument('--eval_file_path',type=str,default="sentiment_output.json")
    parser.add_argument('--eval_batch_size',type=int,default=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    args.device = device
    main(args)



