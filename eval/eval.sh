model_name_or_path=../root/autodl-tmp/model/air/best_sentiment_classifier
device_num=0
dataset_path=../test_data/prefix_dic/Air_sentiment_140.0_256_insert_prefix.jsonl

python eval_sent_acc.py --dataset_path $dataset_path --model_name_or_path "/root/autodl-tmp/model/air/best_sentiment_classifier" --device_num $device_num --acc_log_file $acc_log_file_path --gen_file $json_file