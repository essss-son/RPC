
data=128
length=64
prefix_len=30
prefix_len=20
device_num=0
epochs=10


outputs_path=/root/autodl-tmp/FT/attr/gen/sent_${prefix_len}prefix/2048-mid-dim-${data}data-${epochs}epochs-${length}length_topk200/
outputs_path=/root/autodl-tmp/FT/attr/gen/Gedi/
outputs_path=/root/autodl-tmp/FT/attr2/test_data/txt/
ppl_file=gpt2_medium_imdb_${length}.jsonl
ppl_file=Gedi_medium_toxic_${length}.jsonl
ppl_file=attr_medium_toxic_${length}.jsonl
generated_text0=attr_detoxification_120.0_384.txt 
generated_text1=attr_detoxification_120.0_512.txt 
generated_text0=gpt2_medium_detoxification_attr0_${length}.txt
generated_text1=gpt2_medium_detoxification_attr0_${length}.txt
generated_text0=Dexpert_nontoxic_384.txt
generated_text1=Dexpert_nontoxic_512.txt
generated_text0=GeDi_detoxify_384.txt
generated_text1=GeDi_detoxify_512.txt
generated_text0=jigsaw_nontoxic_gpt2_all_prefix128_ck30000_length_384.txt
generated_text1=jigsaw_nontoxic_gpt2_all_prefix128_ck30000_length_512.txt
json_file=Gedi_medium_toxic_${length}.jsonl
json_file=gpt2_medium_imdb_256_insert_prompt.jsonl
generated_text0=gpt2_medium_imdb_attr0_256.txt
generated_text1=gpt2_medium_imdb_attr1_256.txt
json_file=gpt2_medium_imdb_${length}.jsonl

acc_log_file=acc_log_file.jsonl
acc_log_file_path=$outputs_path$acc_log_file
ppl_log_file=ppl_log_file.jsonl
ppl_log_file_path=$outputs_path$ppl_log_file

python evaluation.py --txt $outputs_path --attr0_txt $generated_text0 --attr1_txt $generated_text1 --ppl_log_file $ppl_log_file_path --gen_file $ppl_file