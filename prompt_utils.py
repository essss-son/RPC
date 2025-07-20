import os

import torch
from transformers import GPT2TokenizerFast, CTRLTokenizer, BartTokenizerFast, Trainer, AutoTokenizer
import random
import numpy as np
import json
from datasets import load_from_disk

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            yield json.loads(line)


def create_prompts(tokenizer_name, prompt_type, control_code=None, generate_eval=False):
     different prompts for eval and test. we use eval prompts to select models
    tokenizer = None
    if tokenizer_name == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('/home/anke/FT/model/openai-community/gpt2')
        tokenizer = GPT2TokenizerFast.from_pretrained('/root/autodl-tmp/model/gpt2-medium')
    elif tokenizer_name == 'bart':
        tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    if prompt_type == 'test':
        prompts = ['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The house', 'The lake',
               'The last time', 'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country',
               'The road', 'The year is 1910.']15
    prompts = ['Once upon a time', 'The book']
    if prompt_type == "topic":
        prompts = ['In summary,', 'This essay discusses', 'Views on', 'The connection', 'Foundational to this is',
                   'To review', 'In brief ', 'An illustration of ', 'Furthermore',
                   'The central theme', 'To conclude', 'The key aspect', 'Prior to this', 'Emphasized are',
                   'To summarize', 'The relationship', 'More importantly', 'It has been shown', 'The issue focused on', 'In this essay']
    if prompt_type == 'json':
        data_path = "/home/anke666/FT/codes/CTG/attr/example/prompt/sentiment_prompts-10k/"
        prompt_file = ['negative_prompts.jsonl', 'neutral_prompts.jsonl', 'positive_prompts.jsonl']
        prompts_all = []
        for f in prompt_file:
            prompt_path = os.path.join(data_path, f)
            sub_prompts = []
            for item in read_jsonl(prompt_path):
                sub_prompts.append(item['prompt']['text'])
            prompts_all.append(sub_prompts)
        prompts_tensors = []
        for i in range(len(prompts_all)):
            sunb_prompts_tensors = []
            half_len = int(len(prompts_all[i]) / 2)
            print("prompts_all")
            print(half_len)
            print(type(prompts_all[i])) list
            for prompt in prompts_all[i][0:half_len]:
                print("prompt")
                print(type(prompt)) str
                half_len = len(prompt) / 2
                sunb_prompts_tensors.append(tokenizer(prompt, return_tensors='pt'))
            prompts_tensors.append(sunb_prompts_tensors)

        return prompts_tensors

    if prompt_type == 'toxic':
        data_path = "/root/autodl-tmp/FT/attr/prompt_detoxification.txt"
        prompts = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                prompts.append(line.strip())
        prompts_tensors = []
        for prompt in prompts:
            prompts_tensors.append(tokenizer(prompt, return_tensors='pt'))
        return prompts_tensors

    if prompt_type == 'prompt':
        prompts = []
        prompts = prompts + ['We all know', 'University Professor', 'Clearly,', 'Canadian film-maker', 'In London,',
                   'This film, released in 1951,', 'No one would ever question', "If you're looking for",
                   'I have seen this movie', "I've never expected ", 'Thursday June 9, 9:15pm', "That's how Burt Reynolds describes",
                   'May 2nd', 'Six months after high-school', 'To paraphrase', 'Once again Woody Allen seems', 'An Italian/American co-production',
                   'An orphan boy named Tom', 'Spanish director', 'Writer & director Jay Andrews,', 'An Asian', 'Well, some people might',
                   'STAR RATING:', 'Someday,', "It's wartime drama - WWII,", 'Does anyone remember', 'At 20 years old, ', 'Today, ',
                   'My Take: Yet another lame', 'Ever notice how so many', "I've heard some", 'Everything everyone has said already',
                   'Like most sports movies which', 'After seeing', 'With its rerelease', 'Early 1950s Sci-Fi directed by',
                   'Being an American service member', 'Our reviewer from Toronto ', 'Three kids are born', 'OK, let me',
                   'A flesh-eating octopus', 'For those viewers who thought', 'One: your screen will be', 'I´m from germany',
                   'After slightly over 50 years', 'What exactly do people expect', 'For those who expect documentaries',
                   'Two college buddies', 'I, like so many others', '"Gargle with old razor blades.', 'Simply put,', 'My ratings:',
                   'My 5 year old daughter', 'Let me start out by', "Perhaps it's just me, ", 'Flesh Feast starts at Miami',
                   'You would probably get something', 'A March 1947 New York', 'As we are well aware,', 'Naturally,', 'A scientist',
                   'A comedy talk', 'Jackie Chan movies', 'So many times, ', "I'm not sure if these", 'It would seem we should',
                   "All right, let's be realistic", 'Scarecrow Gone Wild starts as', 'In Pasadena, ', 'While it comes no closer',
                   'My friend made me', 'Based on its current IMDb', 'There are other reviews here', "I've seen", 'In 1968 when',
                   "Wealthy businessman's daughter,", 'A klutzy young man returns', 'American Tourist on package tour',
                   'A young woman,', '1st watched 10/29/2006 ', "Looking back on Jim Henson's", 'French cinema had always been',
                   'My wife', 'An opera diva', 'There are some things']

    if prompt_type == 'imdb':
        prompts = []
        data_file = '/home/anke666/FT/codes/CTG/attr/datasets_file/sentiment_texts'   数据文件路径，数据需要提前下载
        dataset = load_from_disk(data_file)
        print(dataset)
         print(dataset["train"]["label"])
        dataset = dataset.filter(lambda x: x["text"] is not None)
        dataset_pos = dataset["Positive"]
        dataset_neg = dataset["Negative"]
        examples = dataset_pos["text"] + dataset_neg["text"]
         example = random.choice(examples)
        texts = examples["input_ids"]
        for i in range(len(texts)):
            prompts.append(tokenizer.decode(texts[i][0:4]))
        print(texts)

    if generate_eval:
        prompts = ['Anyone who wishes', 'As of', 'It is', 'In October 2014', 'While people can', 'Not only',
                   'The park', 'The game', 'The reward', 'The paper']10

    if tokenizer_name == 'ctrl':
        assert control_code is not None, 'control_code can not be None'
        final_prompts = []
        for prompt in prompts:
            final_prompts.append(control_code + ' ' + prompt)
        prompts = final_prompts
    prompts_tensors = []
    for prompt in prompts:
        prompts_tensors.append(tokenizer(prompt, return_tensors='pt'))
        attention_mask.append(tokenizer(prompt, return_tensors='pt').attention_mask)

    return prompts_tensors


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
