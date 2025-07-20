import torch
import os
from torch.nn import functional as F
import numpy as np
import project_config
import math
from datasets import Dataset
import random
from tqdm import tqdm

import inspect

def air_generate_insert_prefix(args):
     存储每次决策的动作和log概率
    actions_list = []
    log_probs_list = []
    insert_count_list = []
    rewards = []

     设置最小插入间隔，避免多个同时插入
    insert_interval_count = 32
    f = open("../test_data/attr_{}_{}_128.jsonl".format(args.task_mode, args.lambda_cs), 'w')
    tokenizer = args.tokenizer
    model = args.generation_model
    critic_model = args.critic_model
    times = 1
    clean_res = []
    args.samples = 1
    for i in range(times):
        for type in args.att_type:
            if args.task_mode == "detoxification":
                if type == '1':
                    continue
            for prompt in args.prompt:                
                with torch.no_grad():
                    input_text = torch.tensor([tokenizer(tokenizer.eos_token + prompt).input_ids]).long().to(
                        args.device)
                    max_length = args.max_length
                    past_key_values = None
                    prev = None
                    input_text = input_text.expand(int(args.samples / times), input_text.shape[-1])
                    if args.task_mode != 'detoxification':
                        cur_len = len(tokenizer.encode(prompt))
                    else:
                        cur_len = 0
                    result = input_text[:, input_text.shape[-1] - cur_len:]

                    att_dic = dict()
                    prob = dict()
                    sigma_ij = dict()
                    cond_logits = dict()

                    actions = []
                    log_probs = []
                    insert_count = 0
                    while cur_len < max_length:
                        print(f"cur_len_insert:{cur_len}")
                        
                        insert_sign = False
                        计算插入prefix
                        if cur_len != 0 and cur_len % 32 == 0:
                             当前文本作为Critic的输入
                            outputs_critic = model.transformer(input_ids=prev, past_key_values=past_key_values)
                            lm_hidden_states = outputs_critic.last_hidden_state

                             获取Critic的输出（允许梯度传播）
                            with torch.enable_grad():
                                logits = critic_model(lm_hidden_states)
                                probs = logits.squeeze()
            
                                 定义分类分布并采样动作
                                action_dist = torch.distributions.Categorical(probs)
                                print(f"Probabilities: {action_dist.probs}")
                                action = action_dist.sample()
                                log_prob = action_dist.log_prob(action)
                                actions.append(action)
                                log_probs.append(log_prob)
                                if action.item() == 1:
                                    insert_count += 1
                                    insert_sign = True
                            
                        if past_key_values is None:
                            dic_base = model(input_ids=input_text, return_dict=True, use_cache=True, use_prefix=False, insert_sign=False)
                            logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                            for item in args.att_type:
                                if args.task_mode == "sentiment":
                                    prefix_id = int(item)
                                elif args.task_mode == "topic":
                                    prefix_id = int(item) + 2
                                elif args.task_mode == "detoxification":
                                    prefix_id = int(item) + 6

                                att_dic[item] = dict()
                                att_dic[item]['dict'] = model(input_ids=input_text, return_dict=True, use_cache=True,
                                                              use_prefix=True, prefix_id=prefix_id, insert_sign=insert_sign)
                                att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                                att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values
                                att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                                prob[item] = torch.ones(int(args.samples / times), 1).to(args.device)

                            for item_i in args.att_type:
                                for item_j in args.att_type:
                                    if item_j != item_i:
                                        sigma_ij[item_i + item_j] = prob[item_i] / prob[item_j]

                        else:
                            dic_base = model(input_ids=prev, past_key_values=past_key_values, return_dict=True,
                                             use_cache=True, use_prefix=False, insert_sign=False)
                            logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                            for item in args.att_type:
                                if args.task_mode == "sentiment":
                                    prefix_id = int(item)
                                elif args.task_mode == "topic":
                                    prefix_id = int(item) + 2
                                elif args.task_mode == "detoxification":
                                    prefix_id = int(item) + 6

                                att_dic[item]['dict'] = model(input_ids=prev, past_key_values=att_dic[item]['past_kv'],
                                                              return_dict=True, use_cache=True, use_prefix=True,
                                                              prefix_id=prefix_id, insert_sign=insert_sign)
                                att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                                att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values

                                prob[item] = torch.gather(att_dic[item]['logits_norm'], dim=-1, index=prev)
                                att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                            for item_i in args.att_type:
                                for item_j in args.att_type:
                                    if item_j != item_i:
                                        sigma_ij[item_i + item_j] *= prob[item_i] / prob[item_j]

                        logits_norm_base = torch.softmax(logits_base, dim=-1)

                        for item_i in args.att_type:
                            cond_logits[item_i] = None
                            for item_j in args.att_type:
                                if cond_logits[item_i] == None:
                                    if item_j == item_i:
                                        cond_logits[item_i] = att_dic[item_j]['logits_norm']
                                    else:
                                        cond_logits[item_i] = att_dic[item_j]['logits_norm'] * sigma_ij[item_j + item_i]
                                else:
                                    if item_j == item_i:
                                        cond_logits[item_i] = cond_logits[item_i] + att_dic[item_j]['logits_norm']
                                    else:
                                        cond_logits[item_i] = cond_logits[item_i] + att_dic[item_j]['logits_norm'] * \
                                                              sigma_ij[item_j + item_i]

                            cond_logits[item_i] = att_dic[item_i]['logits_norm'] / cond_logits[item_i]
                            cond_logits[item_i] = torch.nan_to_num(cond_logits[item_i], nan=0)

                        cond_att_logits = cond_logits[type]

                        next_token_logits = logits_norm_base * (cond_att_logits ** args.lambda_cs)

                        top_probs, top_indices = torch.topk(next_token_logits, args.topk, dim=-1)

                        try:
                            tmp_prev = torch.multinomial(top_probs, num_samples=1)
                        except:
                            print("end generation")
                            raise Exception("Too high lambda_cs")
                        prev = top_indices.gather(-1, tmp_prev)
                        if prev == 50256: <endoftext>结束生成
                            break
                        result = torch.cat((result, prev), dim=-1)

                        cur_len = cur_len + 1

                actions_list.append(actions)
                log_probs_list.append(log_probs)
                for i in range(int(args.samples / times)):
                    clean_res.append(tokenizer.decode(result[i])) 两条一条neg，一条pos
    
    return clean_res, actions_list, log_probs_list, insert_count_list

def air_generate(args):
    f = open("../test_data/attr_{}_{}_128.jsonl".format(args.task_mode, args.lambda_cs), 'w')
    tokenizer = args.tokenizer
    model = args.generation_model
    params = inspect.signature(model.forward)
    print(params)
    times = 1
    clean_res = []
    args.samples = 1
    for i in range(times):
        for type in args.att_type:
            if args.task_mode == "detoxification":
                if type == '1':
                    continue
            for prompt in args.prompt:
                with torch.no_grad():
                    input_text = torch.tensor([tokenizer(tokenizer.eos_token + prompt).input_ids]).long().to(
                        args.device)
                    max_length = args.max_length
                    past_key_values = None
                    prev = None
                    input_text = input_text.expand(int(args.samples / times), input_text.shape[-1])
                    if args.task_mode != 'detoxification':
                        cur_len = len(tokenizer.encode(prompt))
                    else:
                        cur_len = 0
                    result = input_text[:, input_text.shape[-1] - cur_len:]

                    att_dic = dict()
                    prob = dict()
                    sigma_ij = dict()
                    cond_logits = dict()
                    insert_sign = False
                    while cur_len < max_length: 
                        print(f"cur_len:{cur_len}")
                        if past_key_values is None:
                            dic_base = model(input_ids=input_text, return_dict=True, use_cache=True, use_prefix=False, insert_sign=False)
                            logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                            for item in args.att_type:
                                if args.task_mode == "sentiment":
                                    prefix_id = int(item)
                                elif args.task_mode == "topic":
                                    prefix_id = int(item) + 2
                                elif args.task_mode == "detoxification":
                                    prefix_id = int(item) + 6

                                att_dic[item] = dict()
                                att_dic[item]['dict'] = model(input_ids=input_text, return_dict=True, use_cache=True,
                                                              use_prefix=True, prefix_id=prefix_id, insert_sign=insert_sign)
                                att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                                att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values
                                att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                                prob[item] = torch.ones(int(args.samples / times), 1).to(args.device)

                            for item_i in args.att_type:
                                for item_j in args.att_type:
                                    if item_j != item_i:
                                        sigma_ij[item_i + item_j] = prob[item_i] / prob[item_j]

                        else:
                            dic_base = model(input_ids=prev, past_key_values=past_key_values, return_dict=True,
                                             use_cache=True, use_prefix=False, insert_sign=False)
                            logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                            for item in args.att_type:
                                if args.task_mode == "sentiment":
                                    prefix_id = int(item)
                                elif args.task_mode == "topic":
                                    prefix_id = int(item) + 2
                                elif args.task_mode == "detoxification":
                                    prefix_id = int(item) + 6

                                att_dic[item]['dict'] = model(input_ids=prev, past_key_values=att_dic[item]['past_kv'],
                                                              return_dict=True, use_cache=True, use_prefix=True,
                                                              prefix_id=prefix_id, insert_sign=insert_sign)
                                att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                                att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values

                                prob[item] = torch.gather(att_dic[item]['logits_norm'], dim=-1, index=prev)
                                att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                            for item_i in args.att_type:
                                for item_j in args.att_type:
                                    if item_j != item_i:
                                        sigma_ij[item_i + item_j] *= prob[item_i] / prob[item_j]

                        logits_norm_base = torch.softmax(logits_base, dim=-1)

                        for item_i in args.att_type:
                            cond_logits[item_i] = None
                            for item_j in args.att_type:
                                if cond_logits[item_i] == None:
                                    if item_j == item_i:
                                        cond_logits[item_i] = att_dic[item_j]['logits_norm']
                                    else:
                                        cond_logits[item_i] = att_dic[item_j]['logits_norm'] * sigma_ij[item_j + item_i]
                                else:
                                    if item_j == item_i:
                                        cond_logits[item_i] = cond_logits[item_i] + att_dic[item_j]['logits_norm']
                                    else:
                                        cond_logits[item_i] = cond_logits[item_i] + att_dic[item_j]['logits_norm'] * \
                                                              sigma_ij[item_j + item_i]

                            cond_logits[item_i] = att_dic[item_i]['logits_norm'] / cond_logits[item_i]
                            cond_logits[item_i] = torch.nan_to_num(cond_logits[item_i], nan=0)

                        cond_att_logits = cond_logits[type]

                        next_token_logits = logits_norm_base * (cond_att_logits ** args.lambda_cs)

                        top_probs, top_indices = torch.topk(next_token_logits, args.topk, dim=-1)

                        try:
                            tmp_prev = torch.multinomial(top_probs, num_samples=1)
                        except:
                            print("end generation")
                            raise Exception("Too high lambda_cs")
                        prev = top_indices.gather(-1, tmp_prev)
                        if prev == 50256: <endoftext>结束生成
                            break
                        result = torch.cat((result, prev), dim=-1)

                        cur_len = cur_len + 1


                for i in range(int(args.samples / times)):
                    clean_res.append(tokenizer.decode(result[i])) 两条一条neg，一条pos
    
    return clean_res






def compute_advantage(attribute_score1, attribute_score2, ppl1, ppl2, alpha=100.0, beta=1.0):
     优势函数设计：对比两种情况的属性分数和困惑度
    print(f"attribute_score1 - attribute_score2:{(attribute_score1 - attribute_score2)}")
    print(f"ppl1 - ppl2:{(ppl1 - ppl2)}")
    reward = []
    for i in range(len(attribute_score1)):
        reward.append(alpha * (attribute_score1[i] - attribute_score2[i]) - beta * (ppl1[i] - ppl2[i]))
    return reward
    
    
def compute_perplexity_per_sample(model, tokenizer, texts, device):
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'], use_prefix=False)

         获取每个 token 的损失，按 mask 计算有效 token 数量
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = encodings['input_ids'][..., 1:].contiguous()

         计算交叉熵损失（逐 token）
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())   恢复为 [batch_size, seq_len] 的形状

         按有效 token 求平均 loss
        attention_mask = encodings['attention_mask'][..., 1:]
        loss_per_sample = (loss * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

     对每条文本计算 PPL
    perplexities = [math.exp(l.item()) for l in loss_per_sample]

    return perplexities



