import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import jsonlines
from tqdm import tqdm

 读取模型和tokenizer
model_name = '/root/autodl-tmp/model/cls/roberta-base-3'   假设你已经训练好的3分类roberta模型
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

 移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

 创建Dataset类
class SentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        
         从jsonl文件读取数据
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                self.texts.append(obj['text'])
                label = obj['sentiment']
                 将标签转换为数字
                if label == "Negative":
                    self.labels.append(0)
                elif label == "Positive":
                    self.labels.append(2)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
         使用tokenizer进行文本编码
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
         返回编码后的文本和标签
        return {
            'input_ids': encoding['input_ids'].squeeze(0),   去掉batch维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

 创建DataLoader用于批量处理
batch_size = 8
dataset = SentimentDataset(file_path='/root/autodl-tmp/FT/air/air/test_data/new_generates/Air_sentiment_140.0_length_64_ck7_JS.jsonl', tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

 准备评估
model.eval()
correct_preds = 0
total_preds = 0

tp = 0   True Positives
fp = 0   False Positives
fn = 0   False Negatives

 逐批次处理
for batch in tqdm(dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
     进行推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

     获取预测结果
    preds = torch.argmax(logits, dim=-1)

     计算 TP, FP, FN
    for i in range(len(preds)):
        if preds[i] == labels[i]:   预测正确
            tp += 1
            if preds[i] == 2:   预测为Positive
                tp1 += 1
            elif preds[i] == 0:   预测为Negative
                tp0 += 1
        else:
            if preds[i] == 2:   预测为Positive，且实际不是
                fp1 += 1
            elif preds[i] == 0:   预测为Negative，且实际不是
                fp0 += 1

     统计准确率
    correct_preds += (preds == labels).sum().item()
    total_preds += labels.size(0)

 计算准确率 (Accuracy)
acc_all = tp / (tp + fp1 + fp0)
acc_0 = tp0 / (tp0 + fp0)
acc_1 = tp1 / (tp1 + fp1)


 输出结果
print(f"Acc_all: {acc_all:.4f}, acc_0: {acc_0:.4f}, acc_1: {acc_1:.4f}")

 计算准确率
accuracy = correct_preds / total_preds
print(f"Accuracy: {accuracy:.4f}")
