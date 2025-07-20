import os
from transformers import pipeline

def evaluate_bart_large_mnli(
    world_file="world.txt",
    sports_file="sports.txt",
    business_file="business.txt",
    sci_tech_file="sci_tech.txt",
    threshold=0.5
):
    """
    使用 facebook/bart-large-mnli 对四个文本文件做零样本多标签分类，
    并计算指定的准确度指标。
    """

     1. 定义候选标签及其对应的索引
    candidate_labels = ["World", "Sports", "Business", "Science", "Technology"]
    
     label -> index 的映射
     注：Science 和 Technology 都映射为 index=3 (sci/tech)
    label_to_idx = {
        "World": 0,
        "Sports": 1,
        "Business": 2,
        "Science": 3,
        "Technology": 3
    }

     2. 准备计数器
     tp[i]：预测到第 i 类且该行文本实际也是第 i 类
     fp[i]：预测到第 i 类但该行文本实际不是第 i 类
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]   该类别的真实样本但未被正确预测的数

     3. 加载 Zero-shot Classification pipeline，指定多标签 & GPU
    classifier = pipeline(
        "zero-shot-classification",
        model="/root/autodl-tmp/model/Facebook/bart-large-mnli",   或本地模型路径
        device=0,                          如果有 GPU，0 表示第一个 GPU；无 GPU 则改为 -1
        multi_label=True
    )

     4. 定义一个函数：对单个文本进行预测，并更新计数器
    def classify_and_update(text: str, true_label_idx: int):
        """
        对一行文本做多标签预测，比较预测结果与真实标签，更新 tp / fp 计数器。
        """
         使用零样本分类
        result = classifier(text, candidate_labels)

         result["labels"] 形如: ["Business","World","Sports","Technology","Science"]
         result["scores"] 形如: [0.91, 0.05, 0.02, 0.01, 0.01]

        pred_labels = []
        for lbl, score in zip(result["labels"], result["scores"]):
            if score >= threshold:
                pred_labels.append(lbl)

         遍历所有被预测到的标签
        for pl in pred_labels:
            pred_idx = label_to_idx[pl]
            if pred_idx == true_label_idx:
                tp[pred_idx] += 1
            else:
                fp[pred_idx] += 1
          统计 FN（真实类别未被预测）
        if true_label_idx not in pred_labels:
            fn[true_label_idx] += 1

     5. 针对不同文件，循环读取文本，更新计数器
     world.txt -> ground_truth_idx = 0
    print("start world")
    if os.path.isfile(world_file):
        with open(world_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    classify_and_update(text, true_label_idx=0)
    else:
        print(f"未找到文件: {world_file}")

     sports.txt -> ground_truth_idx = 1
    print("start sports")
    if os.path.isfile(sports_file):
        with open(sports_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    classify_and_update(text, true_label_idx=1)
    else:
        print(f"未找到文件: {sports_file}")

     business.txt -> ground_truth_idx = 2
    print("start business")
    if os.path.isfile(business_file):
        with open(business_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    classify_and_update(text, true_label_idx=2)
    else:
        print(f"未找到文件: {business_file}")

     sci_tech.txt -> ground_truth_idx = 3
    print("start sci_tech")
     注意：无论预测到 "Science" 还是 "Technology"，都视为预测到 class 3
    if os.path.isfile(sci_tech_file):
        with open(sci_tech_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    classify_and_update(text, true_label_idx=3)
    else:
        print(f"未找到文件: {sci_tech_file}")

     6. 计算准确度指标
     tp_all = sum(tp[i] for i in [0,1,2,3])
     fp_all = sum(fp[i] for i in [0,1,2,3])
    tp_all = sum(tp)
    fp_all = sum(fp)
    fn_all = sum(fn)

     避免分母为 0 的情况
    def safe_div(a, b):
        return a / b if b != 0 else 0

    acc = safe_div(tp_all, tp_all + fp_all)
    rec = safe_div(tp_all, tp_all + fn_all)   Recall
    f1 = safe_div(2 * acc * rec, acc + rec)   F1-score
    
    acc0 = safe_div(tp[0], tp[0] + fp[0])
    acc1 = safe_div(tp[1], tp[1] + fp[1])
    acc2 = safe_div(tp[2], tp[2] + fp[2])
    acc3 = safe_div(tp[3], tp[3] + fp[3])
     Per-class
    acc_per_class = [safe_div(tp[i], tp[i] + fp[i]) for i in range(4)]
    rec_per_class = [safe_div(tp[i], tp[i] + fn[i]) for i in range(4)]
    f1_per_class = [
        safe_div(2 * acc_per_class[i] * rec_per_class[i], acc_per_class[i] + rec_per_class[i])
        for i in range(4)
    ]

     7. 打印评估结果
    print("===== 评估结果 =====")
    print(f"TP = {tp}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP_all = {tp_all}, FP_all = {fp_all}, FN_all = {fn_all}")

    print(f"Overall Precision (acc) = {acc:.4f}")
    print(f"Overall Recall (rec)    = {rec:.4f}")
    print(f"Overall F1-score        = {f1:.4f}")

    print(f"Per-class Precision:")
    print(f"  acc0 (World)     = {acc_per_class[0]:.4f}, rec0 = {rec_per_class[0]:.4f}, F1_0 = {f1_per_class[0]:.4f}")
    print(f"  acc1 (Sports)    = {acc_per_class[1]:.4f}, rec1 = {rec_per_class[1]:.4f}, F1_1 = {f1_per_class[1]:.4f}")
    print(f"  acc2 (Business)  = {acc_per_class[2]:.4f}, rec2 = {rec_per_class[2]:.4f}, F1_2 = {f1_per_class[2]:.4f}")
    print(f"  acc3 (Sci/Tech)  = {acc_per_class[3]:.4f}, rec3 = {rec_per_class[3]:.4f}, F1_3 = {f1_per_class[3]:.4f}")
if __name__ == "__main__":
    evaluate_bart_large_mnli(
        world_file="/root/autodl-tmp/FT/air/test_data/rand/air_world_64.txt",
        sports_file="/root/autodl-tmp/FT/air/test_data/rand/air_sports_64.txt",
        business_file="/root/autodl-tmp/FT/air/test_data/rand/air_business_64.txt",
        sci_tech_file="/root/autodl-tmp/FT/air/test_data/rand/air_science_64.txt",
        threshold=0.5
    )
