import torch
import argparse
import torch.nn as nn
import torch.optim as optim


import sys
import inspect
sys.path.append('/root/autodl-tmp/FT/attr2')   确保路径正确

from modeling_gpt2_2 import PrefixGPT2LMHeadModel
from model_sent import RobertaForPreTraining


import modeling_gpt2
print(modeling_gpt2.__file__)
print("*"*200)
from transformers import GPT2Tokenizer, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import os
from model_utils import *
from utils import *
from datasets import load_from_disk, concatenate_datasets
import jsonlines
from torch.nn import functional as F

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="使用强化学习训练Critic网络辅助受控文本生成")
    
     模型路径参数
    parser.add_argument('--generation_model_name', type=str, default='gpt2-medium', help='预训练生成模型名称或路径')
    parser.add_argument('--attribute_classifier_path', type=str, required=True, help='预训练属性分类器路径')
    parser.add_argument('--datasets_path', type=str, required=True, help='数据集路径')
    
     训练相关参数
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--max_length', type=int, default=256, help='生成文本的最大长度')
    parser.add_argument('--task_type', type=int, default=0, help='任务类型，情感：0，主题：1，去毒：2')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    parser.add_argument("--lambda_cs", default=140.0, type=float, help="control strength")
    parser.add_argument("--topk", default=200, type=int)


    parser.add_argument("--task_mode", default='sentiment', type=str, choices=['sentiment', 'topic', 'detoxification'])
    parser.add_argument("--att_type", default=['0', '1'],
                        help='sentiment:0 in positive and 1 in negative; topic: 0 in world, 1 in sports, 2 in business, and 3 in science; detoxification: 0 in nontoxic and 1 in toxic')
    
     保存路径
    parser.add_argument('--save_path', type=str, default='./critic_model', help='训练好的Critic模型保存路径')
    
    args = parser.parse_args()
    
     强制batch_size为1
    if args.batch_size != 1:
        print("警告: 由于生成过程中的动态prefix插入，batch_size已被强制设置为1。")
        args.batch_size = 1

    set_seed(args)
    
    return args

def tokenize(sample):
    txt_in_len = 4
    topic = sample["topic"]
    print(topic)
    if topic == 0:
        target_attribute = 0
    else:
        target_attribute = 2
    print(sample["input_ids"])
    print(len(sample["input_ids"]))
    sample["input_ids"] = sample["input_ids"][:txt_in_len]
    print(sample["input_ids"])
    print(len(sample["input_ids"]))
    sample["target_attribute"] = target_attribute
    return sample

def padding_fuse_fn(data_list):
    txt_in_len = 4
    input_ids = []
    target_attribute = []
    for i, item in enumerate(data_list):
        text = item["input_ids"][:txt_in_len]
        if item["topic"] == 0:
            target_attribute.append(0)
        else:
            target_attribute.append(2)

        input_ids.append(text)

    batch = {}
    batch["input_ids"] = input_ids
    batch["target_attribute"] = target_attribute

    return batch


 使用Critic网络生成文本
def generate_text_with_critic(args, device, task_type=0, max_length=256):
    args.generation_model.eval()
    args.critic_model.train()   设置为训练模式以允许梯度计算

    if task_type == 0:
        target_attribute = [1, 0]
    elif task_type == 1:
        target_attribute = [0, 1, 2, 3]
    elif task_type == 2:
        target_attribute = [0]
    device = args.device
    
     对prompt进行编码
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(input_ids)
    print(input_ids.size())
   
    wop_score = []
    wp_score = []
    
    
    不插入的样本
    sent_wop = air_generate(generation_model, tokenizer, generated_ids, cur_token, past_key_values, max_length , device)  这里生成完整的样本
    sent_wop_list = air_generate(args)  这里返回两条文本，一条neg，一条pos对于情感，对于主题就有4个
    用分类器算一下score
    print(f"start to generate wop")
    wop_encodings = args.attribute_classifier_tokenizer(
            sent_wop_list, 
            return_tensors='pt', 
            truncation=True, 
            padding='longest'   动态填充到当前 batch 中最长句子的长度
            ).to(device)
    wop_encodings['attention_mask'] = wop_encodings['attention_mask'].unsqueeze(1).unsqueeze(2)
    wop_encodings['attention_mask'] = (1.0 - wop_encodings['attention_mask']) * -10000.0
    with torch.no_grad():
        wop_logits = args.attribute_classifier(input_ids=wop_encodings['input_ids'], attention_mask=wop_encodings['attention_mask'])
        print(f"wop_logits:{wop_logits}")
        print(f"wop_logits:{wop_logits}")
        for i in range(len(target_attribute)):
            wop_score.append(wop_logits[i, target_attribute[i]])    
    print(f"wop_score:{wop_score}")
    wop_ppl = compute_perplexity_per_sample(args.generation_model, args.tokenizer, sent_wop_list, device)
    print(f"wop_ppl:{wop_ppl}")

    print(f"start to generate wp")
    sent_wp_list, actions_list, log_probs_list, insert_count_list = air_generate_insert_prefix(args)
    wp_encodings = args.attribute_classifier_tokenizer(
            sent_wp_list, 
            return_tensors='pt', 
            truncation=True, 
            padding='longest'   动态填充到当前 batch 中最长句子的长度
            ).to(device)
    wp_encodings['attention_mask'] = wp_encodings['attention_mask'].unsqueeze(1).unsqueeze(2)
    wp_encodings['attention_mask'] = (1.0 - wp_encodings['attention_mask']) * -10000.0
    with torch.no_grad():
        wp_logits = args.attribute_classifier(input_ids=wp_encodings['input_ids'], attention_mask=wp_encodings['attention_mask'])
        print(f"wp_logits:{wp_logits}")
        print(f"wop_logits:{wop_logits}")
        for i in range(len(target_attribute)):
            wp_score.append(wp_logits[i, target_attribute[i]])
    print(f"wp_score:{wp_score}")
    wp_ppl = compute_perplexity_per_sample(args.generation_model, args.tokenizer, sent_wp_list, device)
    print(f"wp_ppl:{wp_ppl}")

    advantage = compute_advantage(wp_score, wop_score, wp_ppl, wop_ppl, alpha=100.0, beta=1)
    rewards = advantage

    return sent_wop_list, sent_wp_list, actions_list, log_probs_list, rewards, insert_count_list

 主训练函数
def main():
    args = parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    
     创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
     加载分词器
    args.tokenizer = GPT2Tokenizer.from_pretrained("/root/autodl-tmp/model/gpt2-medium")
    args.tokenizer.pad_token = args.tokenizer.eos_token   GPT-2默认没有pad token
    
     加载生成模型和Prefix模型
    args.generation_model = PrefixGPT2LMHeadModel.from_pretrained(args.generation_model_name)
    params = inspect.signature(args.generation_model.forward)
    print(params)
    
     实例化Prefix模型
    prefix_model = PrefixModel.from_pretrained('').to(device)
    
     冻结生成模型参数
    for param in args.generation_model.parameters():
        param.requires_grad = False
    args.generation_model.to(device)
    
     加载Critic网络
    args.critic_model = CriticNetwork(n_embd=args.generation_model.transformer.embed_dim, num_labels=2).to(device)
    
     加载属性分类器
    args.attribute_classifier = AttributeClassifier(pretrained_model=args.attribute_classifier_path).to(device)
    args.attribute_classifier = RobertaForPreTraining.from_pretrained(args.attribute_classifier_path)
    args.attribute_classifier.to(args.device)
    args.attribute_classifier.eval()
    args.attribute_classifier_tokenizer = RobertaTokenizer.from_pretrained(args.attribute_classifier_path)

    
     准备数据集（此处使用示例数据，实际应用中请使用真实数据集）
    dataset = load_from_disk(args.datasets_path)
    train_dataset = dataset['train']
    pos_dataset = train_dataset.select(range(0,1000))
    neg_dataset = train_dataset.select(range(20000,21000))
    merged_dataset = concatenate_datasets([pos_dataset, neg_dataset])
    merged_dataset = merged_dataset.map(tokenize, batched=False)
    dataloader = DataLoader(merged_dataset, collate_fn=padding_fuse_fn, batch_size=args.batch_size, shuffle=True)
    
     定义优化器
    optimizer = optim.AdamW(args.critic_model.parameters(), lr=args.lr)

    
     训练循环
    for epoch in range(args.epochs):
        args.critic_model.train()
        total_loss = 0

        accumulated_loss = 0
        accumulated_samples = 0
        step = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            step += 1
            print(batch)
            input_ids = batch['input_ids']
            print(input_ids)
            print(input_ids)
            args.prompt = args.tokenizer.batch_decode(input_ids)
            print(f"the length of prompt_list:{len(args.prompt)}")
            print(args.prompt)
             由于batch_size=1，取一个样本            
             使用Critic网络生成文本
            sent_wop, generated_text, actions_list, log_probs_list, rewards, insert_count_list = generate_text_with_critic(args,
                device, task_type=args.task_type,  max_length=args.max_length
            )

                    
             计算困惑度
             perplexity = compute_perplexity(generation_model, tokenizer, generated_text, device)
            if step % 10 == 0:
                print(sent_wop)
                print("+" * 100)
                print(generated_text)
            
             计算奖励
            print(f"rewards:{rewards}")
            try:
                rewards = torch.stack(rewards).squeeze()
            except:
                continue
            if not rewards:
                continue
            
             计算损失：策略梯度损失，单独每个动作乘以奖励
            for i in range(len(log_probs_list)):
                try:
                    log_probs_tensor = torch.stack(log_probs_list[i])
                    print(f"log_probs_tensor:{log_probs_tensor}")
                    loss = - (log_probs_tensor * rewards[i]).sum()
                except:
                    continue
                print(loss)
            
                 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                total_loss += loss.item()
            
                 累加损失和样本数
                accumulated_loss += loss.item()
                accumulated_samples += 1

                if accumulated_samples >= 8 and accumulated_loss > 0:
                    optimizer.step()   更新参数
                    optimizer.zero_grad()   清零梯度
        
                     重置累积损失和样本数
                    accumulated_loss = 0
                    accumulated_samples = 0
            break
            with jsonlines.open(args.save_path + f'/insert_count_file_{epoch}.jsonl', mode="a") as file_jsonl:
                for i in range(len(actions_list)):
                    for j in range(len(actions_list[i])):
                        actions_list[i][j] = actions_list[i][j].tolist()
                item = {
                    "epoch": epoch,
                    "step": step,
                    "insert_count_list": insert_count_list,
                    "actions_list": actions_list,
                }
                file_jsonl.write(item)
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss")        

    
         保存训练好的Critic网络
        torch.save(critic_model.state_dict(), os.path.join(args.save_path, 'critic_model.pt'))
        save_path = args.save_path + f'/epoch-{epoch}'
        critic_model.save_pretrained(save_path)
        print("Critic模型已保存。")

if __name__ == "__main__":
    main()
