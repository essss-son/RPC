import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from datasets import load_from_disk
from torch import nn

 假设我们使用的是一个与训练时相同的模型结构（例如 RoBERTa 分类模型）
 这里的 RoBERTa 模型假设是二分类任务
class Discriminator(nn.Module):
    def __init__(self, model_name="/root/autodl-tmp/model/cls/roberta-base-3/"):   RoBEsRTa model for sentiment analysis
        super(Discriminator, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def forward(self, input_text):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        print(inputs)
        print(inputs["input_ids"].size())
        print(inputs["attention_mask"].size())
        logits = self.model(**inputs).logits
        print(logits)
        out_logits = torch.gather(logits, dim=-1, index=sentiment)
        print(out_logits)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 1. 定义判别器模型结构
discriminator = RobertaForSequenceClassification.from_pretrained("/root/autodl-tmp/model/cls/roberta-base-3/")
discriminator = RobertaForSequenceClassification.from_pretrained("/root/autodl-tmp/model/air/best_sentiment_classifier/")
tokenizer = RobertaTokenizer.from_pretrained("/root/autodl-tmp/model/cls/roberta-base-3/")

model = Discriminator()
 2. 加载模型的权重
discriminator.load_state_dict(torch.load('/root/autodl-tmp/attr2/gan/seqgan_discriminator_roberta.pth'))
model.load_state_dict(torch.load('/root/autodl-tmp/attr2/gan/seqgan_discriminator_roberta.pth'))

 3. 设置为评估模式
discriminator.to(device)
discriminator.eval()
model.to(device)
model.eval()


data_file = '/root/autodl-tmp/datasets/imdb_sentiment_gpt2_mix'
dataset = load_from_disk(data_file)
test_dataset = dataset["test"]
print(len(test_dataset))

 假设你有一个测试数据集
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_labels = []
all_preds = []

 不需要计算梯度
with torch.no_grad():
    for batch in test_loader:
    for i in range(int(len(test_dataset) / 8)):
        texts = test_dataset["text"][i * 8 : (i + 1) * 8]
        labels = test_dataset["topic"][i * 8 : (i + 1) * 8]
        print(labels)
        for j in range(len(texts)):
            texts[j] = texts[j].strip('\n')
            if labels[j] == 1:
                labels[j] = 2
        labels = torch.tensor(labels)
        print(labels)   
        print(texts)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
         获取判别器的输出
        logits = discriminator(**inputs).logits
        logits = model(texts)
        print(logits)

         判别器输出的是未归一化的 logits, 使用 sigmoid 进行二分类
        probs = torch.sigmoid(logits)
        print(probs)
        

         将概率转为标签，阈值为 0.5
        preds = torch.argmax(probs, dim=1)
        preds = (probs > 0.5).long()
        print(preds)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        break

 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')

 如果需要更详细的分类评估
from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds))

