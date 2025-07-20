export CUDA_LAUNCH_BLOCKING=1
lm_model_path=/root/autodl-tmp/FT/attr2/ckpt/ckpt-7-prefixlen-20-bs-4-epoch-5
cls_model_path=/root/autodl-tmp/model/best_sentiment_classifier
datasets_path=/root/autodl-tmp/datasets/imdb_sentiment_gpt2_mix_128

python train_rl.py \
    --generation_model_name $lm_model_path \
    --attribute_classifier_path $cls_model_path \
    --datasets_path $datasets_path\
    --epochs 5 \
    --batch_size 1 \
    --lr 1e-5 \
    --max_length 256 \
    --lambda_cs 140.0 \
    --save_path ./critic_model_100_1/prefix
