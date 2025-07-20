model_name_or_path='/root/autodl-tmp/model/air/best_topic_classifier'
device_num=0
dataset_path='/root/autodl-tmp/FT/air/air/test_data/new_generates/Air_topic_60.0_length_512.jsonl'

python ../eval_topic_acc.py --dataset_path $dataset_path --model_name_or_path $model_name_or_path --device_num $device_num
