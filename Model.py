import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification,GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Model
import torch.nn.functional as F
from torch.distributions import Categorical

class policy(nn.Module):
    def __init__(self,state_dim,action_dim,device=torch.device('cpu')):
        super(policy, self).__init__()
        self.policy_model = GPT2Model.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("/home/anke/DXZ/models/gpt2-medium")
        self.linear = nn.Linear(state_dim,action_dim).to(device)
        self.action_dim = action_dim
        self.device = device

    def forward(self,input_ids):
        encoded = input_ids.to(self.device)
        hidden_state = self.policy_model(input_ids=encoded).last_hidden_state[0] # note 取0是因为batch_size为1
        # size hidden_state: [seq_len, embed_dim]
        logits = self.linear(hidden_state[-1])
        # size output: [2]
        return logits

    def get_action(self,input_ids):
        logits = self.forward(input_ids)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action

    def get_log_prob(self,input_ids, action):
        logits = self.forward(input_ids)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)

        return log_prob


if __name__ == '__main__':
    model = policy(1024,2)

    text = "I love you"
    action = model.get_action(text)
    print(action)
    print(model.get_log_prob(text, action))