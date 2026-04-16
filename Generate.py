import torch
import os
from utils import Agent
import argparse
from tqdm import tqdm
def generate(args):
    agent = Agent(args)
    agent.policy.load_state_dict(torch.load(args.policy_model_path))
    print("model loaded")

    if os.path.exists(args.output_path):
        pass
    else:
        os.makedirs(args.output_path)
    # file = os.path.join(args.output_path, f"{args.task_mode}_output.json")
    # args.output_path = file
    for i in tqdm(range(args.epoch)):
        _, _, insert_count_dict = agent.collect_trajectory_with_policy()
        pos_insert_count = insert_count_dict['0']
        neg_insert_count = insert_count_dict['1']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--length',type=int,default=384)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--lambda_cs',type=float,default=140.0)
    parser.add_argument('--insert_prob',type=float,default=None,help="test only")
    parser.add_argument('--task_mode',type=str,default="sentiment")

    parser.add_argument('--output_path',type=str,default="./output/",help="path of trajectory texts")
    parser.add_argument('--policy_model_path',type=str,default="/home/anke/DXZ/RPC/rpc-new/policy_model/policy_ckpt_28.80.pt")


    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    task_att = dict()
    task_att['sentiment'] = dict()
    task_att['sentiment']['0'] = 'Positive'
    task_att['sentiment']['1'] = 'Negative'
    task_att['topic'] = dict()
    task_att['topic']['0'] = 'World'
    task_att['topic']['1'] = 'Sports'
    task_att['topic']['2'] = 'Business'
    task_att['topic']['3'] = 'Science'
    task_att['detoxification'] = dict()
    task_att['detoxification']['0'] = 'nontoxic'
    task_att['detoxification']['1'] = 'toxic'
    args.task_att = task_att
    generate(args)


