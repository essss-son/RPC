model_name_or_path=../root/autodl-tmp/model/air/best_sentiment_classifier
device_num=0
dataset_path=../test_data/new_generates/Air_sentiment_140.0_length_64_ck7_JS.jsonl

python ../eval_sent_acc.py --dataset_path $dataset_path --model_name_or_path "/root/autodl-tmp/model/air/best_sentiment_classifier" --device_num $device_num