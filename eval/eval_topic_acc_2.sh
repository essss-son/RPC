device_num=0

txt_path=/root/autodl-tmp/FT/attr2/test_data/txt/
json_path=/root/autodl-tmp/FT/attr2/test_data/
acc_log_file=acc_log_file.jsonl
ppl_log_file=ppl_log_file.jsonl
acc_log_file_path=$txt_path$acc_log_file
ppl_log_file_path=$txt_path$ppl_log_file



!/bin/bash
 定义遍历的 length 值
lengths=(64 128 192 256 384 512)

 遍历不同的 length 值
for length in "${lengths[@]}"
do
     设置路径和文件名
    ppl_file=attr_topic_${length}   
    ppl_file=attr_topic_60.0_${length}.jsonl
    json_file=attr_topic_60.0_${length}.jsonl
    json_file_path=$json_path$json_file

    generated_text0=attr_world_60.0_${length}.txt
    generated_text1=attr_sports_60.0_${length}.txt
    generated_text2=attr_business_60.0_${length}.txt
    generated_text3=attr_science_60.0_${length}.txt



     执行评估
    echo "Evaluating with length = $length"
    
    python evaluation_topic.py --txt $txt_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --attr2_txt $generated_text2 --attr3_txt $generated_text3 --txt_len $length --ppl_log_file $ppl_log_file_path --gen_file $ppl_file

python txt2json_topic.py --gen_dir_name $outputs_path --World_txt $generated_text0 --Sports_txt $generated_text1 --Business_txt $generated_text2 --Science_txt $generated_text3 --json_file_name $json_file


    python eval_topic_acc.py --dataset_path $json_file_path --model_name_or_path "/root/autodl-tmp/model/air/best_topic_classifier" --device_num $device_num --acc_log_file $acc_log_file_path --gen_file $json_file
    
    echo "Finished evaluation for length = $length"
done

