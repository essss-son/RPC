device_num=0
output_dir=../ckpt
python ../train_PCLMs.py --model_name_or_path "/root/autodl-tmp/model/gpt2-large" --output_dir $output_dir --device_num $device_num --batch_size 2


samples=50
task_mode=sentiment
lambda_cs=140.0
device_num=0

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/model/air/ckpt_for_sentiment_and_topic"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 64

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 128

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 192

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 256

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 384

python ../air_decoding_insert.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 512



task_mode=topic
lambda_cs=60.0


python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 64

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 128

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 192

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --samples $samples --length 256