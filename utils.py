import torch
import torch.nn.functional as F

import os.path
import random
from transformers import GPT2Tokenizer
from Modeling_gpt2 import GPT2LMHeadModel
import numpy as np
import torch
from Model import policy
from torch.optim import Adam
import json


class Agent:
    def __init__(self,args):
        self.args = args
        self.device = args.device
        self.lambda_cs = args.lambda_cs
        self.policy = policy(state_dim=1024,action_dim=2,device=self.device)
        self.lr = args.lr
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)


        self.task_mode = args.task_mode
        if self.task_mode == "sentiment":
            self.att_type = ['0', '1']
        elif self.task_mode == "topic":
            self.att_type = ['0', '1', '2', '3']
        elif self.task_mode == "detoxification":
            self.att_type = ['0', '1']

        self.tokenizer = GPT2Tokenizer.from_pretrained('/home/anke/DXZ/models/gpt2-medium')
        self.base_model = GPT2LMHeadModel.from_pretrained('/home/anke/DXZ/RPC/Air-Decoding-main/models/ckpt_for_sentiment_and_topic').to(self.device)
        self.batch_size = args.batch_size
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.task_att = args.task_att

    def reset(self):
        prompts = [
            "Once upon a time",
            "The book",
            "The chicken",
            "The city",
            "The country",
            "The horse",
            "The lake",
            "The last time",
            "The movie",
            "The painting",
            "The pizza",
            "The potato",
            "The president of the country",
            "The road",
            "The year is 1910."
        ]
        prompt = random.choice(prompts)
        self.prompt = prompt


    def list_init(self):
        mask = []
        action_list = []
        log_prob_list = []
        reward_list = []
        return mask, action_list, log_prob_list, reward_list

    def collect_trajectory_with_policy(self):
        if self.args.insert_prob is not None:
            insert_prob = self.args.insert_prob
            probs = torch.tensor([1 - insert_prob, insert_prob],device=self.device)
        else:
            probs = None

        self.base_model.eval()
        text_dict = {}
        self.reset()
        if self.args.output_path is not None:
            if os.path.exists(self.args.output_path):
                pass
            else:
                os.makedirs(self.args.output_path)
            # file_path = os.path.join(self.args.output_path, f'{self.task_mode}_result.json')

        return_dict_ = {}
        insert_count_dict = {}
        for type in self.att_type:
            mask, action_list, log_prob_list, reward_list = self.list_init()
            insert_count = 0
            if self.task_mode == "detoxification":
                if type == '1':
                    continue
            input_text = torch.tensor([self.tokenizer(self.tokenizer.eos_token + self.prompt).input_ids]).long().to(self.device)
            max_length = self.args.length
            past_key_values = None
            prev = None
            input_text = input_text.expand(self.batch_size, input_text.shape[-1])
            if self.task_mode != 'detoxification':
                cur_len = len(self.tokenizer.encode(self.prompt))
            else:
                cur_len = 0
            result = input_text[:, input_text.shape[-1] - cur_len:]

            att_dic = dict()
            prob = dict()
            sigma_ij = dict()
            cond_logits = dict()

            while cur_len < max_length:
                if past_key_values is None:
                    # note 生成第一个token
                    dic_base = self.base_model(input_ids=input_text, return_dict=True, use_cache=True, use_prefix=False)
                    logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                    action = self.policy.get_action(input_ids=input_text)
                    log_prob = self.policy.get_log_prob(input_ids=input_text, action=action)

                    action_list.append(action)
                    log_prob_list.append(log_prob)

                    if cur_len % 32 == 0:
                        if self.args.insert_prob is not None:
                            # insert_signal = bool(action.item())
                            insert_signal = bool(probs.multinomial(1).item())
                        else:
                            insert_signal = bool(action.item())
                        if insert_signal == False:
                            pass
                        else:
                            insert_count += 1
                        mask += [1]


                    else:
                        insert_signal = False
                        mask += [0]

                    for item in self.att_type:
                        if self.task_mode == "sentiment":
                            prefix_id = int(item)
                        elif self.task_mode == "topic":
                            prefix_id = int(item) + 2
                        elif self.task_mode == "detoxification":
                            prefix_id = int(item) + 6

                        att_dic[item] = dict()
                        att_dic[item]['dict'] = self.base_model(input_ids=input_text, return_dict=True, use_cache=True,
                                                      use_prefix=True, prefix_id=prefix_id, insert=insert_signal,insert_count=insert_count)
                        att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                        att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values
                        att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                        prob[item] = torch.ones(self.batch_size, 1).to(self.device)

                    for item_i in self.att_type:
                        for item_j in self.att_type:
                            if item_j != item_i:
                                sigma_ij[item_i + item_j] = prob[item_i] / prob[item_j]

                else:
                    dic_base = self.base_model(input_ids=prev, past_key_values=past_key_values, return_dict=True,
                                     use_cache=True, use_prefix=False)
                    logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                    action = self.policy.get_action(input_ids=input_text)
                    log_prob = self.policy.get_log_prob(input_ids=input_text, action=action)

                    action_list.append(action)
                    log_prob_list.append(log_prob)

                    if cur_len % 32 == 0:
                        if self.args.insert_prob is not None:
                            # insert_signal = bool(action.item())
                            insert_signal = bool(probs.multinomial(1).item())
                        else:
                            insert_signal = bool(action.item())
                        if insert_signal == False:
                            pass
                        else:
                            insert_count += 1
                        mask += [1]
                    else:
                        insert_signal = False
                        mask += [0]

                    for item in self.att_type:
                        if self.task_mode == "sentiment":
                            prefix_id = int(item)
                        elif self.task_mode == "topic":
                            prefix_id = int(item) + 2
                        elif self.task_mode == "detoxification":
                            prefix_id = int(item) + 6

                        att_dic[item]['dict'] = self.base_model(input_ids=prev, past_key_values=att_dic[item]['past_kv'],
                                                      return_dict=True, use_cache=True, use_prefix=True,
                                                      prefix_id=prefix_id, insert=insert_signal,insert_count=insert_count)
                        att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                        att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values

                        prob[item] = torch.gather(att_dic[item]['logits_norm'], dim=-1, index=prev)
                        att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                    for item_i in self.att_type:
                        for item_j in self.att_type:
                            if item_j != item_i:
                                sigma_ij[item_i + item_j] *= prob[item_i] / prob[item_j]

                # logits_norm_base = torch.softmax(att_dic[type]['logits'], dim=-1)
                logits_norm_base = torch.softmax(logits_base, dim=-1)

                for item_i in self.att_type:
                    cond_logits[item_i] = None
                    for item_j in self.att_type:
                        if cond_logits[item_i] is None:
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

                next_token_logits = logits_norm_base * (cond_att_logits ** self.lambda_cs)

                top_probs, top_indices = torch.topk(next_token_logits, 200, dim=-1)

                try:
                    tmp_prev = torch.multinomial(top_probs, num_samples=1)
                except:
                    raise Exception("Too high lambda_cs")
                prev = top_indices.gather(-1, tmp_prev)
                # size prev:[batch_size, 1]
                result = torch.cat((result, prev), dim=-1)
                # size result:[batch_size, seq_len]
                # reward = reward_cal(result) # TODO
                # reward_list.append(reward)
                cur_len = cur_len + 1

            clean_res = []
            for i in range(self.batch_size):
                clean_res.append(self.tokenizer.decode(result[i]))
            text_dict[type] = clean_res
            if self.args.output_path is not None:
                if self.args.insert_prob is not None:
                    file_path = self.args.output_path + f"{self.task_mode}_output_{self.args.insert_prob}_{self.args.length}.json"
                else:
                    file_path = self.args.output_path + f"{self.task_mode}_output_{self.args.length}.json"
                with open(file_path, mode='a', encoding='utf-8') as f:
                    if self.task_mode != 'detoxification':
                        for i, text in enumerate(clean_res):
                            data = {}
                            data['text'] = text
                            data[self.task_mode] = self.task_att[self.task_mode][type]
                            data['insert_count'] = insert_count
                            f.write(json.dumps(data) + '\n')
                    else:
                        data = dict()
                        data['prompt'] = self.prompt
                        data['text'] = dict()
                        for i, text in enumerate(clean_res):
                            data['text'][i] = text
                        f.write(json.dumps(data) + '\n')

            return_dict_[type] = {
                'mask':torch.tensor(mask,dtype=torch.float, device=self.device),
                'action_list':torch.tensor(action_list,dtype=torch.float, device=self.device),
                # 'log_prob_list':torch.tensor(log_prob_list,dtype=torch.float, device=self.device),
                'log_prob_list':torch.stack(log_prob_list,dim=0)
            }
            insert_count_dict[type] = insert_count

        return text_dict, return_dict_, insert_count_dict

    def collect_trajectory(self):
        self.base_model.eval()
        self.reset()
        text_dict = {}
        for type in self.att_type:
            if self.task_mode == "detoxification":
                if type == '1':
                    continue
            with torch.no_grad():
                input_text = torch.tensor([self.tokenizer(self.tokenizer.eos_token + self.prompt).input_ids]).long().to(
                    self.device)
                max_length = self.args.length
                past_key_values = None
                prev = None
                input_text = input_text.expand(self.batch_size, input_text.shape[-1])
                if self.task_mode != 'detoxification':
                    cur_len = len(self.tokenizer.encode(self.prompt))
                else:
                    cur_len = 0
                result = input_text[:, input_text.shape[-1] - cur_len:]

                att_dic = dict()
                prob = dict()
                sigma_ij = dict()
                cond_logits = dict()
                while cur_len < max_length:
                    if past_key_values is None:
                        dic_base = self.base_model(input_ids=input_text, return_dict=True, use_cache=True, use_prefix=False)
                        logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                        for item in self.att_type:
                            if self.task_mode == "sentiment":
                                prefix_id = int(item)
                            elif self.task_mode == "topic":
                                prefix_id = int(item) + 2
                            elif self.task_mode == "detoxification":
                                prefix_id = int(item) + 6

                            att_dic[item] = dict()
                            att_dic[item]['dict'] = self.base_model(input_ids=input_text, return_dict=True, use_cache=True,
                                                          use_prefix=True, prefix_id=prefix_id)
                            att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                            att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values
                            att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                            prob[item] = torch.ones(self.batch_size, 1).to(self.device)

                        for item_i in self.att_type:
                            for item_j in self.att_type:
                                if item_j != item_i:
                                    sigma_ij[item_i + item_j] = prob[item_i] / prob[item_j]

                    else:
                        dic_base = self.base_model(input_ids=prev, past_key_values=past_key_values, return_dict=True,
                                         use_cache=True, use_prefix=False)
                        logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                        for item in self.att_type:
                            if self.task_mode == "sentiment":
                                prefix_id = int(item)
                            elif self.task_mode == "topic":
                                prefix_id = int(item) + 2
                            elif self.task_mode == "detoxification":
                                prefix_id = int(item) + 6

                            att_dic[item]['dict'] = self.base_model(input_ids=prev, past_key_values=att_dic[item]['past_kv'],
                                                          return_dict=True, use_cache=True, use_prefix=True,
                                                          prefix_id=prefix_id)
                            att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                            att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values

                            prob[item] = torch.gather(att_dic[item]['logits_norm'], dim=-1, index=prev)
                            att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                        for item_i in self.att_type:
                            for item_j in self.att_type:
                                if item_j != item_i:
                                    sigma_ij[item_i + item_j] *= prob[item_i] / prob[item_j]

                    logits_norm_base = torch.softmax(logits_base, dim=-1)

                    for item_i in self.att_type:
                        cond_logits[item_i] = None
                        for item_j in self.att_type:
                            if cond_logits[item_i] is None:
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

                    next_token_logits = logits_norm_base * (cond_att_logits ** self.lambda_cs)

                    top_probs, top_indices = torch.topk(next_token_logits, 200, dim=-1)

                    try:
                        tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    except:
                        raise Exception("Too high lambda_cs")
                    prev = top_indices.gather(-1, tmp_prev)
                    result = torch.cat((result, prev), dim=-1)

                    cur_len = cur_len + 1
                clean_res = []
                for i in range(self.batch_size):
                    clean_res.append(self.tokenizer.decode(result[i]))

            text_dict[type] = clean_res
        # note result_dict一共 2 * batch_size 条句子,情感任务里
        return text_dict

    def policy_update(self):
        insert_text_dict, return_dict,_ = self.collect_trajectory_with_policy()
        text_dict = self.collect_trajectory()

        rewards = sent_reward_cal(
            result=text_dict,
            insert_result=insert_text_dict,
            sent_classifier=self.args.sent_classifier,
            sent_tokenizer=self.args.sent_tokenizer,
            ppl_model=self.args.ppl_model,
            ppl_tokenizer=self.args.ppl_tokenizer,
            device=self.device,
            alpha=self.args.alpha,
            beta=self.args.beta,
        )
        rewards = torch.stack(rewards, dim=0) # size [2] 先positive 后negative
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        returns = []

        for i,reward in enumerate(rewards):
            returns.append(reward.unsqueeze(0).expand(return_dict[str(i)]['mask'].shape[0],))
        total_loss = torch.tensor(0., device=self.device)
        for att_dict, return_ in zip(return_dict.values(), returns):
            num = att_dict['mask'].sum()
            loss = -(att_dict['log_prob_list'] * att_dict['mask'] * (return_.detach())).sum() / num
            total_loss += loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), rewards.mean().item()


def sent_reward_cal(result,
               insert_result,
               sent_classifier,#disbert
               sent_tokenizer, #disbert
               ppl_model, #gpt2medium
               ppl_tokenizer, #gpt2tokenizer
               device,
               alpha=10.0, beta=0.1):
    # note 这里只考虑情感分类，主题需要其他合适的分类器
    model_path = "/home/anke/DXZ/models/distilbert-base-uncased-finetuned-sst-2-english"
    pos_insert_text = insert_result['0']
    neg_insert_text = insert_result['1']
    pos_text = result['0']
    neg_text = result['1']
    # 各batch_size条 将batch_size条相同情感的分数求平均作为一个prompt对应的分数

    with torch.no_grad():
        try:
            pos_insert_score = sentiment_cal(pos_insert_text, sent_tokenizer, sent_classifier, "positive", device)
            neg_insert_score = sentiment_cal(neg_insert_text, sent_tokenizer, sent_classifier, "negative", device)
            pos_score = sentiment_cal(pos_text, sent_tokenizer, sent_classifier, "positive", device)
            neg_score = sentiment_cal(neg_text, sent_tokenizer, sent_classifier, "negative", device)
        except Exception as e:
            print(f"error: {e}")
            print(f"result: {result}")
            print(f"insert_result: {insert_result}")
            tmp = sent_tokenizer(result, return_tensors='pt', padding=True)
            insert_tmp = sent_tokenizer(insert_result, return_tensors='pt', padding=True)
            print(f"tmp.size:{tmp.input_ids.shape}")
            print(f"insert_tmp.size:{insert_tmp.input_ids.shape}")
            raise
        pos_insert_ppl = cal_ppl(ppl_model,pos_insert_text,ppl_tokenizer,device )
        neg_insert_ppl = cal_ppl(ppl_model,neg_insert_text,ppl_tokenizer,device)
        pos_ppl = cal_ppl(ppl_model,pos_text,ppl_tokenizer,device)
        neg_ppl = cal_ppl(ppl_model,neg_text,ppl_tokenizer,device)

    insert_score_list = [pos_insert_score,neg_insert_score]
    score_list = [pos_score,neg_score]

    insert_ppl_list = [pos_insert_ppl,neg_insert_ppl]
    ppl_list = [pos_ppl,neg_ppl]
    reward = []
    for i_score,score, i_ppl, ppl in zip(insert_score_list,score_list,insert_ppl_list,ppl_list):
        s = alpha * (i_score - score) - beta * (i_ppl - ppl)
        reward.append(s)

    return reward
def sentiment_cal(texts, tokenizer,classifier,sentiment, device):
    # tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(texts, return_tensors='pt', padding=True).to(device)
    logits = classifier(**encoded).logits
    probs = F.softmax(logits, dim=-1)
    # size batch_size, 2
    score = probs.mean(dim=0)

    if sentiment == "positive":
        return score[1]
    elif sentiment == "negative":
        return score[0]

def cal_ppl(model,texts,tokenizer, device):
    #这里tokenizer是gpt2 tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(texts, return_tensors='pt', padding=True).to(device)
    mask = encoded['attention_mask'][:,1:]
    # note 是否要对有padding的情况单独进行考虑？
    # input_ids = encoded.input_ids[:,:-1]
    target_ids = encoded.input_ids[:,1:]
    logits = model(**encoded).logits
    probs = F.softmax(logits, dim=-1)[:,:-1,:]
    selected_ids = torch.gather(probs, dim=-1, index=target_ids.unsqueeze(dim=-1)).squeeze(dim=-1)
    # size selected_ids:[batch_size, seq_len]
    ppl = torch.exp(-((torch.log(selected_ids) * mask).sum(dim=1)) / mask.sum(dim=1)).mean(dim=0)
    return ppl

if __name__ == '__main__':
    pass
