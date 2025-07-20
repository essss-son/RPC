length=128
device_num=0
outputs_path=/root/autodl-tmp/FT/attr2/eval/data/2/2/

ppl_file=GeDi__${length}.jsonl

generated_text0=GeDi_world_${length}.txt 
generated_text1=GeDi_sports_${length}.txt
generated_text2=GeDi_business_${length}.txt 
generated_text3=GeDi_science_${length}.txt

json_file=GeDi_topic_${length}.jsonl
acc_log_file=acc_log_file_${length}.jsonl
acc_log_file_path=$outputs_path$acc_log_file
ppl_log_file=ppl_log_file_${length}.jsonl
ppl_log_file_path=$outputs_path$ppl_log_file

python json2txt_topic.py --gen_dir_name $outputs_path --World_txt $generated_text0 --Sports_txt $generated_text1 --Business_txt $generated_text2 --Science_txt $generated_text3 --json_file_name $json_file

python evaluation_topic.py --txt $outputs_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --attr2_txt $generated_text2 --attr3_txt $generated_text3 --ppl_log_file $ppl_log_file_path --gen_file $ppl_file

json_file_path=$outputs_path$json_file

python eval_topic_acc.py --dataset_path $json_file_path --model_name_or_path "/root/autodl-tmp/model/air/best_topic_classifier" --device_num $device_num --acc_log_file $acc_log_file_path --gen_file $json_file