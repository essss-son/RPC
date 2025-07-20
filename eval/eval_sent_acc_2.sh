device_num=0
txt_path=/root/autodl-tmp/FT/attr2/test_data/finetune/txt/
json_path=/root/autodl-tmp/FT/attr2/test_data/finetune/
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
    ppl_file=attr_sentiment_${length}
    generated_text0=attr_neg_120.0_${length}.txt 
    generated_text1=attr_pos_120.0_${length}.txt
    json_file=prefix_0_sentiment_${length}.jsonl
    json_file_path=$json_path$json_file

     执行评估
    echo "Evaluating with length = $length"

    python evaluation.py --txt $txt_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --ppl_log_file $ppl_log_file_path --gen_file $ppl_file

    python eval_sent_acc.py --dataset_path $json_file_path --model_name_or_path "/root/autodl-tmp/model/air/best_sentiment_classifier" --device_num $device_num --acc_log_file $acc_log_file_path --gen_file $json_file

    echo "Finished evaluation for length = $length"
done

