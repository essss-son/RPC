import json
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
files = [
    "/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_128_.json",
    "/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_256_.json",
    "/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_384_.json",
    "/home/anke/DXZ/RPC/rpc-new/output/sentiment_output_512_.json"
]
cnts = [np.arange(1,4),np.arange(1,7), np.arange(1,10),np.arange(1,13)]
i_cnts = []
for file,cnt in zip(files,cnts):
    plt.figure()
    probs = np.random.rand(len(cnt))
    probs /= probs.sum()
    i_cnt = np.random.choice(cnt, 512, p=probs).tolist()
    cnts = Counter(i_cnt)

    l = sorted(cnts.items(),key=lambda x: x[0])
    x = np.array([x[0] for x in l])
    plt.xticks(x,x)
    plt.xlabel("Insert count")
    y = np.array([x[1] for x in l])
    plt.bar(x,y)
    length = int(file.split('/')[-1][-9:-6])
    plt.savefig(f"policy_insert_length_{length}.png")
    plt.close()