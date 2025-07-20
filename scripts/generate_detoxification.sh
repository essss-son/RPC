samples=10
task_mode=detoxification
lambda_cs=120.0
device_num=0

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 64

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 128

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384

python ../air_decoding.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5" --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512