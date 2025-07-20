 !/opt/conda/bin/python3
 -*- coding: utf-8 -*-

import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from datasets import load_metric
import operator
from functools import reduce
import project_config
import argparse
import jsonlines

def geometric_mean(values):
    return (reduce(operator.mul, values)) ** (1.0 / len(values))



def create_dataset_from_txt(txt):
    assert txt is not None
    with open(txt, 'r', encoding='utf-8-sig') as rf:
        lines = rf.readlines()
    sents = []
    print(f"lines:{len(lines)}")
    for sent in lines:
        if len(sent) < 3:
            continue
        sent.strip('"')
        sents.append(sent.replace('\n', ''))
    print(f"sents:{len(sents)}")
    return sents


def compute_ppl(model, sents):
    dataloader = DataLoader(sents, batch_size=1, shuffle=False, drop_last=False)
    tokenizer = GPT2TokenizerFast.from_pretrained('/root/autodl-tmp/model/gpt2-medium')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    lst_ppl = []
    tqdm_iter = tqdm(dataloader, desc='compute ppl')
    for data in dataloader:
        input = tokenizer(data, return_tensors='pt')
        input_ids = input['input_ids'].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        lst_ppl.append(torch.exp(outputs[0]).item())
    avg_ppl = np.around(np.mean(lst_ppl), 2)
    return avg_ppl


def compute_distinct(sents):
    word_lst = []
    total_sent = ' '.join(sents)
    total_sent = total_sent.split(' ')
    len_total_sent = len(total_sent)
    for i in range(len_total_sent):
        word_lst.append(total_sent[i])
    gram_1_lst = list(set(word_lst))
    word_2_lst = []
    word_3_lst = []
    word_4_lst = []
    for sent in sents:
        sent = sent.split(' ')
        len_sent = len(sent)
        for i in range(len_sent-1):
            word_2_lst.append(sent[i] + ' ' + sent[i+1])
    for sent in sents:
        sent = sent.split(' ')
        len_sent = len(sent)
        for i in range(len_sent-2):
            word_3_lst.append(sent[i] + ' ' + sent[i+1] + ' ' + sent[i+2])
    for sent in sents:
        sent = sent.split(' ')
        len_sent = len(sent)
        for i in range(len_sent-3):
            word_4_lst.append(sent[i] + ' ' + sent[i+1] + ' ' + sent[i+2] + ' ' + sent[i+3])
    gram_2_lst = list(set(word_2_lst))
    gram_3_lst = list(set(word_3_lst))
    gram_4_lst = list(set(word_4_lst))
    dis1 = round(len(gram_1_lst) / len(word_lst), 4)
    dis2 = round(len(gram_2_lst) / len(word_2_lst), 4)
    dis3 = round(len(gram_3_lst) / len(word_3_lst), 4)
    dis4 = round(len(gram_4_lst) / len(word_4_lst), 4)
    return {'distinct1': dis1,
            'distinct2': dis2,
            'distinct3': dis3,
            'distinct4': dis4,
            }


def distinctness(generations):
    dist1, dist2, dist3 = [], [], []
     calculate dist1, dist2, dist3 across generations for every prompt
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    dist1.append(len(unigrams) / total_words)
    dist2.append(len(bigrams) / total_words)
    dist3.append(len(trigrams) / total_words)


     take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


def geometric_mean(values):
    return (reduce(operator.mul, values)) ** (1.0 / len(values))


def main(args):
    txt_list = []
    world_txt = args.txt + f'gpt2_medium_imdb_World_{args.txt_len}_insert_prefix.txt'
    sports_txt = args.txt + f'gpt2_medium_imdb_Sports_{args.txt_len}_insert_prefix.txt'
    business_txt = args.txt + f'gpt2_medium_imdb_Business_{args.txt_len}_insert_prefix.txt'
    science_txt = args.txt + f'gpt2_medium_imdb_SciTech_{args.txt_len}_insert_prefix.txt'
    world_txt = args.txt + args.attr0_txt
    sports_txt = args.txt + args.attr1_txt
    business_txt = args.txt + args.attr2_txt
    science_txt = args.txt + args.attr3_txt
    txt_list.append(world_txt)
    txt_list.append(sports_txt)
    txt_list.append(business_txt)
    txt_list.append(science_txt)

    assert txt_list is not None
    ppl_list=[]
    dis_list=[]
    for i in range(len(txt_list)):
        txt_file = txt_list[i]
        print("[Evaluating results from {}]".format(txt_file))
        sents = create_dataset_from_txt(txt_file)   list

        ppl = {}

        for model in ['gpt2', 'gpt2-medium', 'gpt2-large']:  
            base_model = "/root/autodl-tmp/model/" + model
            gpt2 = GPT2LMHeadModel.from_pretrained(base_model).cuda()
            print(len(sents))
            current_ppl = compute_ppl(gpt2, sents)
            ppl[model] = current_ppl
        ppl['avg'] = (ppl['gpt2'] + ppl['gpt2-medium'] + ppl['gpt2-large']) / 3

        distinct = compute_distinct(sents)
        dist1, dist2, dist3 = distinctness(sents)

        print(ppl)
        print(distinct)
        print(f"dist-1:{dist1},dist-2:{dist2},dist-3：{dist3}")

        ppl_list.append(ppl)
        dis_list.append(distinct)

    
    with jsonlines.open(args.ppl_log_file, mode="a") as file_jsonl:
        item = {
            "file": args.gen_file,
            "ppl0": ppl_list[0],
            "ppl1": ppl_list[1],
            "ppl2": ppl_list[2],
            "ppl3": ppl_list[3],
            "distinct0": dis_list[0],
            "distinct1": dis_list[1],
            "distinct2": dis_list[2],
            "distinct3": dis_list[3],
        }
        file_jsonl.write(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, default="/root/autodl-tmp/attr2/test_gen/g11252/", help='txt file contains text to be evaluated')
    parser.add_argument('--txt', type=str, default="/root/autodl-tmp/CTG/gen_out/", help='txt file contains text to be evaluated')
    parser.add_argument("--attr0_txt", default="gpt2_medium_imdb_attr0.txt", type=str)
    parser.add_argument("--attr1_txt", default="gpt2_medium_imdb_attr1.txt", type=str)
    parser.add_argument("--attr2_txt", default="gpt2_medium_imdb_attr2.txt", type=str)
    parser.add_argument("--attr3_txt", default="gpt2_medium_imdb_attr3.txt", type=str)
    parser.add_argument("--ppl_log_file", default="gpt2_medium_imdb_attr0.txt", type=str)
    parser.add_argument("--txt_len", default=128, type=int)
    parser.add_argument("--gen_file", default="gpt2_medium_imdb_attr1.txt", type=str)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in evaluating')
    parser.add_argument('--no_cuda', action='store_true', help='evaluate on CPU')
    args = parser.parse_args()
    print(args.txt)
    main(args)
