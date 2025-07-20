model_name_or_path= "/root/model/air/ckpt_for_sentiment_and_topic"
samples=20
task_mode=sentiment
lambda_cs=140.0
device_num=0

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/model/air/ckpt_for_sentiment_and_topic"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64 --samples $samples

python ../air_decoding_insert2.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/model/air/ckpt_for_sentiment_and_topic"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples

python ../air_decoding_insert2.py --model_name_or_path "/root/autodl-tmp/model/air/ckpt_for_sentiment_and_topic"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples 

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256 --samples $samples

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples

python ../gen_insert_prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples



python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 128 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples

