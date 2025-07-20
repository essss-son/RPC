from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification
from datasets import load_dataset,load_from_disk
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import torch
from tqdm import tqdm
import os
from torch.nn import functional as F
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))   Safety check
         Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

         Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
             Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
         Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

         scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def generate(model, input_ids, cur_len, max_length, do_sample, temperature, top_k,
             repetition_penalty, batch_size, eos_token_ids=None,):
    unfinished_sents = input_ids.new(batch_size).fill_(1)
     print("beginning")
     print(unfinished_sents)
     print()
    sent_lengths = input_ids.new(batch_size).fill_(max_length)
    with torch.no_grad():
        cur_token = input_ids
        device = input_ids.device
        past = None
        past_length = 0

        while cur_len < max_length:
             specify position ids (starts with 0)
            input_shape = cur_token.size()
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            past_length += input_shape[-1]

            lm_outputs = model.transformer(cur_token, past_key_values=past, position_ids=position_ids)
            last_hidden_state, past = lm_outputs["last_hidden_state"], lm_outputs["past_key_values"]
            logits = model.lm_head(last_hidden_state)
            print(logits.shape) 2,1,50257

            next_token_logits = logits[:, -1, :]   the last 4 tokens are special tokens, not considered during generation

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                         if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty
            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                 Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k)

                 Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
                print("next_token")
                print(next_token.shape)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

              update generations and finished sentences
             if eos_token_ids is not None:
                  pad finished sentences if eos_token_ids exist
                 tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
             else:
                 tokens_to_add = next_token
            tokens_to_add = next_token
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_token = tokens_to_add.unsqueeze(-1)

            if eos_token_ids is not None:
                eos_token_id = eos_token_ids
                eos_in_sents = tokens_to_add == eos_token_id
                 if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                print(is_sents_unfinished_and_token_to_add_is_eos)
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                 unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

             stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            cur_len = cur_len + 1

    decoded = input_ids
    for hypo_idx, hypo in enumerate(input_ids):
        decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

    return decoded

 加载预训练模型和分词器
model_name = '/root/autodl-tmp/gpt2/gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

data_file = '/root/autodl-tmp/datasets/imdb_sentiment_gpt2_mix'
dataset = load_from_disk(data_file)
train_dataset = dataset["train"]
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=1, pin_memory=True
)


samples_per_batch = 8
sents_generated = []

for epoch in range(1):
    tq = tqdm(train_dataloader, desc='Iteration')
    for step, batch in enumerate(tq):
        input_ids = batch['input_ids'][:6]
        input_ids = torch.tensor(input_ids).cuda()
        cur_len = input_ids.size()[0]
        print(cur_len)
        for i in range(1):
            output = generate(model,
                              input_ids=input_ids.repeat(samples_per_batch, 1).to(device),
                              cur_len=cur_len,
                              max_length=128,
                              do_sample=True,
                              temperature=1,
                              top_k=100,
                              repetition_penalty=1.2,
                              batch_size=samples_per_batch,
                              eos_token_ids=50256,
                              )
            sents = tokenizer.batch_decode(output, skip_special_tokens=True)
            sents_generated.extend(sents)
    dir_name = '/root/autodl-tmp/attr2/test_gen'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    out_file = os.path.join(dir_name, f'pretrain_data.txt')
    with open(out_file, 'w', encoding='utf-8') as wf:
        for sent in sents_generated:
            sent = sent.replace('\n', ' ')
            wf.write(sent + '\n')

