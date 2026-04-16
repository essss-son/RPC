import torch

from Modeling_gpt2 import GPT2LMHeadModel
import os
sent_topic_model = GPT2LMHeadModel.from_pretrained('/home/anke/DXZ/RPC/Air-Decoding-main/models/ckpt_for_sentiment_and_topic')

param = {}
for i in range(24):
    attn = sent_topic_model.transformer.h[i].attn
    param['layer' + str(i)] = {
        "prefix_keys_embeddings_n0": {k: v for k, v in attn.prefix_keys_embeddings_n0.state_dict().items()},
        "prefix_keys_embeddings_n1": {k: v for k, v in attn.prefix_keys_embeddings_n1.state_dict().items()},
        "prefix_keys_embeddings_n2": {k: v for k, v in attn.prefix_keys_embeddings_n2.state_dict().items()},
        "prefix_keys_embeddings_n3": {k: v for k, v in attn.prefix_keys_embeddings_n3.state_dict().items()},
        "prefix_keys_embeddings_n4": {k: v for k, v in attn.prefix_keys_embeddings_n4.state_dict().items()},
        "prefix_keys_embeddings_n5": {k: v for k, v in attn.prefix_keys_embeddings_n5.state_dict().items()},
        "prefix_keys_embeddings_n6": {k: v for k, v in attn.prefix_keys_embeddings_n6.state_dict().items()},
        "prefix_keys_embeddings_n7": {k: v for k, v in attn.prefix_keys_embeddings_n7.state_dict().items()},

        "prefix_values_embeddings_n0": {k: v for k, v in attn.prefix_values_embeddings_n0.state_dict().items()},
        "prefix_values_embeddings_n1": {k: v for k, v in attn.prefix_values_embeddings_n1.state_dict().items()},
        "prefix_values_embeddings_n2": {k: v for k, v in attn.prefix_values_embeddings_n2.state_dict().items()},
        "prefix_values_embeddings_n3": {k: v for k, v in attn.prefix_values_embeddings_n3.state_dict().items()},
        "prefix_values_embeddings_n4": {k: v for k, v in attn.prefix_values_embeddings_n4.state_dict().items()},
        "prefix_values_embeddings_n5": {k: v for k, v in attn.prefix_values_embeddings_n5.state_dict().items()},
        "prefix_values_embeddings_n6": {k: v for k, v in attn.prefix_values_embeddings_n6.state_dict().items()},
        "prefix_values_embeddings_n7": {k: v for k, v in attn.prefix_values_embeddings_n7.state_dict().items()},

        "prefix_mlp_n0":{k: v for k, v in attn.prefix_mlp_n0.state_dict().items()},
        "prefix_mlp_n1":{k: v for k, v in attn.prefix_mlp_n1.state_dict().items()},
        "prefix_mlp_n2":{k: v for k, v in attn.prefix_mlp_n2.state_dict().items()},
        "prefix_mlp_n3":{k: v for k, v in attn.prefix_mlp_n3.state_dict().items()},
        "prefix_mlp_n4":{k: v for k, v in attn.prefix_mlp_n4.state_dict().items()},
        "prefix_mlp_n5":{k: v for k, v in attn.prefix_mlp_n5.state_dict().items()},
        "prefix_mlp_n6":{k: v for k, v in attn.prefix_mlp_n6.state_dict().items()},
        "prefix_mlp_n7":{k: v for k, v in attn.prefix_mlp_n7.state_dict().items()},
    }

if os.path.exists("./prefix_weight/"):
    pass
else:
    os.makedirs("./prefix_weight/")
save_path = "./prefix_weight/" + "prefix_ckpt.pt"

torch.save(param, save_path)
print("保存成功")


