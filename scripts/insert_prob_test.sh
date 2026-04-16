#len=(128 256 384 512)
#bsz=(4 2 1 1)
#epc=(64 128 256 256)
#
#for i in "${!len[@]}"; do
#  python Generate.py \
#    --length "${len[$i]}" \
#    --batch_size "${bsz[$i]}" \
#    --epoch "${epc[$i]}"
#done

#eval acc

fp=(
"/home/anke/DXZ/RPC/rpc-new/output_/sentiment_output_0.5_256.json"
"/home/anke/DXZ/RPC/rpc-new/output_/sentiment_output_0.7_256.json"
"/home/anke/DXZ/RPC/rpc-new/output_/sentiment_output_0.9_256.json"
"/home/anke/DXZ/RPC/rpc-new/output_/sentiment_output_1.0_256.json"
#"/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_128_.json"
#"/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_256_.json"
#"/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_384_.json"
#"/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_512_.json"
)
ev_bsz=(8 8 8 8)
for i in "${!fp[@]}"; do
  python Eval/eval_sent_acc.py \
    --dataset_path "${fp[$i]}" \
    --batch_size "${ev_bsz[$i]}"

  python Eval/eval_perplexity.py \
    --dataset_path "${fp[$i]}" \

  python Eval/eval_dist.py \
    --dataset_path "${fp[$i]}"
done


#eval ppl
#for i in "${!fp[@]}"; do
#  python Eval/Eval_ppl.py \
#    --eval_file_path "${fp[$i]}" \
#    --batch_size "${ev_bsz[$i]}"
#done