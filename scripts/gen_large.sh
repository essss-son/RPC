model_name_or_path= "/root/model/air/ckpt_for_sentiment_and_topic"
samples=50
task_mode=sentiment
lambda_cs=130.0
device_num=0

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/model/air/ckpt_for_sentiment_and_topic"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64 --samples $samples

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 128 --samples $samples

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256 --samples $samples

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples

python ../gen_insert_prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples


task_mode=topic
lambda_cs=60.0
device_num=0

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 128
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 192
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 256

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 128 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples
python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples

