dataset_path=../test_data/Air_sentiment_140.0_length_512.jsonl
model_name_or_path="/root/autodl-tmp/model/cls/roberta-large"

python ../eval_dist.py --dataset_path $dataset_path --model_name_or_path $model_name_or_path