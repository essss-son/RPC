model_name_or_path="/root/autodl-tmp/gpt2/gpt2-medium"
dataset_path=../test_data/new_generates/Air_sentiment_140.0_length_512.jsonl
device_num=0

python ../eval_perplexity.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --device_num $device_num