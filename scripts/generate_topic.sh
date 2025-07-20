model_name_or_path= "/root/model/air/ckpt_for_sentiment_and_topic"

device_num=0

samples=20
task_mode=sentiment
lambda_cs=140.0
device_num=0

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 64

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 128

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 192

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 256

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 384

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 512



task_mode=topic
lambda_cs=60.0
device_num=0



python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 512

samples=20
task_mode=detoxification
lambda_cs=120.0
device_num=0

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 128

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_1/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512
