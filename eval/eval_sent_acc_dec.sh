
length=64

device_num=0




outputs_path=/root/autodl-tmp/FT/attr2/gen/datg/${length}/

acc_log_file=acc_log_file_dec.jsonl
acc_log_file_path=$outputs_path$acc_log_file
ppl_log_file=ppl_log_file_dec.jsonl
ppl_log_file_path=$outputs_path$ppl_log_file

lambda_cs=140.0
ppl_file=gpt2_medium_imdb_${lambda_cs}_${length}
generated_text0=attr_neg_${lambda_cs}_${length}.txt
generated_text1=attr_pos_${lambda_cs}_${length}.txt
generated_text0=DATG_P_NEG.txt 
generated_text1=DATG_P_POS.txt 
json_file=Air_sentiment_${lambda_cs}_${length}_insert_prefix.jsonl
json_file=datg_sentiment_${length}_insert.jsonl

json_outputs_path=/root/autodl-tmp/FT/attr2/gen/datg/${length}/
json_file_path=$json_outputs_path$json_file

python evaluation.py --txt $outputs_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --ppl_log_file $ppl_log_file_path --gen_file $ppl_file
python txt2json.py --gen_dir_name $outputs_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --json_file_name $json_file
python eval_sent_acc.py --dataset_path $json_file_path --model_name_or_path "/root/autodl-tmp/model/air/best_sentiment_classifier" --device_num $device_num --acc_log_file $acc_log_file_path --gen_file $json_file








