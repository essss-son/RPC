

length=64
prefix_len=20
device_num=0
epochs=10

outputs_path=/root/autodl-tmp/FT/attr2/large/
outputs_path=/root/autodl-tmp/FT/attr/gen/De/imdb_196-384/
outputs_path=/home/anke666/FT/codes/CTG/RMT/main/
ppl_file=pplm_imdb_${length}.jsonl
ppl_file=RMT_${length}_insert_prefix.jsonl
ppl_file=GeDi_imdb_${length}.jsonl
generated_text0=gpt2_medium_imdb_attr0_${length}.txt
generated_text1=gpt2_medium_imdb_attr1_${length}.txt
generated_text0=pplm_imdb_neg_${length}.txt
generated_text1=pplm_imdb_pos_${length}.txt
generated_text0=GeDi_imdb_neg_${length}.txt
generated_text1=GeDi_imdb_pos_${length}.txt
generated_text0=rmt_neg_${length}.txt
generated_text1=rmt_pos_${length}.txt
json_file=pplm_imdb.jsonl
json_file=GeDi_imdb.jsonl
json_file=RMT_${length}_insert_prefix.jsonl
generated_text0=gpt2_medium_imdb_attr0_256.txt
generated_text1=gpt2_medium_imdb_attr1_256.txt
json_file=gpt2_medium_imdb_${length}.jsonl

acc_log_file=acc_log_file.jsonl
acc_log_file_path=$outputs_path$acc_log_file
ppl_log_file=ppl_log_file.jsonl
ppl_log_file_path=$outputs_path$ppl_log_file

python evaluation.py --txt $outputs_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --ppl_log_file $ppl_log_file_path --gen_file $ppl_file


python txt2json.py --gen_dir_name $outputs_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --json_file_name $json_file

json_file_path=$outputs_path$json_file

python eval_sent_acc.py --dataset_path $json_file_path --model_name_or_path "/root/autodl-tmp/model/air/best_sentiment_classifier" --device_num $device_num --acc_log_file $acc_log_file_path --gen_file $json_file



