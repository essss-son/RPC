import os
import random
import numpy as np
import torch
from utils import Agent
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.tensorboard import SummaryWriter
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def train(args):
    agent = Agent(args)
    best_reward = -10000

    for episode in tqdm(range(100), desc="Training"):
        loss, reward = agent.policy_update()
        # agent.collect_trajectory()
        args.writer.add_scalar("loss", loss, episode)
        args.writer.add_scalar("reward", reward, episode)
        if reward > best_reward:
            best_reward = reward
            if os.path.exists(args.model_save_path):
                pass
            else:
                os.makedirs(args.model_save_path)
            model_path = os.path.join(args.model_save_path, f"policy_ckpt_{reward:.2f}.pt")
            print(f"model saved, reward:{best_reward}")
            ckpt = agent.policy.state_dict()
            torch.save(ckpt, model_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--accumulation_step', type=int, default=1)
    parser.add_argument('--lambda_cs', type=float, default=140.0)
    parser.add_argument('--alpha', type=float, default=100.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--task_mode', type=str, default="sentiment")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=4,help='num_sequence for per prompt')
    parser.add_argument('--insert_prob',type=float,default=None,help="test only")

    parser.add_argument('--output_path', type=str, default=None,help="path of trajectory texts")
    parser.add_argument('--model_save_path', type=str, default='./policy_model/')

    set_seed(1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.sent_classifier = AutoModelForSequenceClassification.from_pretrained(
        "/home/anke/DXZ/models/distilbert-base-uncased-finetuned-sst-2-english"
    ).to(args.device)
    args.sent_tokenizer = AutoTokenizer.from_pretrained(
        "/home/anke/DXZ/models/distilbert-base-uncased-finetuned-sst-2-english"
    )

    args.ppl_model = GPT2LMHeadModel.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(args.device)
    args.ppl_tokenizer = GPT2Tokenizer.from_pretrained("/home/anke/DXZ/models/gpt2-medium")

    task_att = dict()
    task_att['sentiment'] = dict()
    task_att['sentiment']['0'] = 'Positive'
    task_att['sentiment']['1'] = 'Negative'
    task_att['topic'] = dict()
    task_att['topic']['0'] = 'World'
    task_att['topic']['1'] = 'Sports'
    task_att['topic']['2'] = 'Business'
    task_att['topic']['3'] = 'Science'
    task_att['detoxification'] = dict()
    task_att['detoxification']['0'] = 'nontoxic'
    task_att['detoxification']['1'] = 'toxic'
    args.task_att = task_att


    writer = SummaryWriter(log_dir="./logs/")
    args.writer = writer

    train(args)


