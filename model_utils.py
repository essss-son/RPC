import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification
import math
import os
import json

 定义Prefix模型
class PrefixModel(nn.Module):
    def __init__(self, prefix_size, embedding_dim):
        super(PrefixModel, self).__init__()
         创建一个新的嵌入层
        self.prefix_embedding = nn.Embedding(prefix_size, embedding_dim)
        self.prefix_size = prefix_size
        self.embedding_dim = embedding_dim
    
    def forward(self, input_ids, model):
         获取prefix的嵌入并与输入拼接
        prefix_embeds = self.prefix_embedding(torch.arange(self.prefix_size).to(input_ids.device))
        prefix_embeds = prefix_embeds.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        
         从外部传入的model对象获取wte嵌入层
        input_embeds = model.transformer.wte(input_ids)   GPT-2的token嵌入层
        return torch.cat((prefix_embeds, input_embeds), dim=1)

    def save_pretrained(self, save_directory):
        """
        Saves the model's state_dict and configuration to the specified directory.
        """
         保存模型权重
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
         保存 state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
         保存配置文件（可选，你可以保存一些额外的超参数信息）
        config = {
            "prefix_size": self.prefix_size,
            "embedding_dim": self.embedding_dim,
             可以继续添加其他需要保存的超参数
        }
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        print(f"Model saved in {save_directory}")
        
    @classmethod
    def from_pretrained(cls, save_directory):
        """
        Loads the model's state_dict and configuration from the specified directory.
        """
         加载配置文件
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
         使用配置文件中的参数初始化模型
        prefix_size = config["prefix_size"]
        embedding_dim = config["embedding_dim"]
         使用其他配置参数初始化模型...

         创建模型实例
        model = cls(prefix_size=prefix_size, embedding_dim=embedding_dim)
        
         加载模型权重
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path))
        
        print(f"Model loaded from {save_directory}")
        
        return model

 定义Critic网络
class CriticNetwork(nn.Module):
    def __init__(self, n_embd, num_labels):
        super(CriticNetwork, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
        self.n_embd = n_embd
        self.num_labels = num_labels
        self.score = nn.Linear(self.n_embd, self.num_labels)
    
    def forward(self, hidden_state):
        outputs = self.score(hidden_state)
        return torch.softmax(outputs, dim=-1)   输出概率分布
        
    def save_pretrained(self, save_directory):
        """
        Saves the model's state_dict and configuration to the specified directory.
        """
         保存模型权重
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
         保存 state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
         保存配置文件（可选，你可以保存一些额外的超参数信息）
        config = {
            "n_embd": self.n_embd, 
            "num_labels" : self.num_labels,
             可以继续添加其他需要保存的超参数
        }
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        print(f"Model saved in {save_directory}")
        
    @classmethod
    def from_pretrained(cls, save_directory):
        """
        Loads the model's state_dict and configuration from the specified directory.
        """
         加载配置文件
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
         使用配置文件中的参数初始化模型
        n_embd = config["n_embd"]
        num_labels = config["num_labels"]
         使用其他配置参数初始化模型...

         创建模型实例
        model = cls(n_embd=n_embd, num_labels=num_labels)
        
         加载模型权重
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path))
        
        print(f"Model loaded from {save_directory}")
        
        return model

 定义属性分类器（二分类器）
class AttributeClassifier(nn.Module):
    def __init__(self, pretrained_model='bert-base-3'):
        super(AttributeClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        print(f"bert_out:{outputs.logits}")
        return torch.softmax(outputs.logits, dim=-1)   输出概率分布