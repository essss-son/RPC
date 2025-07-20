model_name_or_path= "/root/model/air/ckpt_for_sentiment_and_topic"
samples=50
task_mode=sentiment
lambda_cs=140.0
device_num=0

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256 --samples $samples --task_type '0'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples --task_type '0'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples --task_type '0'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256 --samples $samples --task_type '0'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples --task_type '0'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples --task_type '0'
python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256 --samples $samples --task_type '1'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt_2/ckpt-7-prefixlen-20-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples --task_type '1'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 192 --samples $samples --task_type '1'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 256 --samples $samples --task_type '1'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 384 --samples $samples --task_type '1'

python ../prefix.py --model_name_or_path "/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-30-bs-4-epoch-5"  --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num --length 512 --samples $samples --task_type '1'

task_mode=topic
lambda_cs=60.0
device_num=0



