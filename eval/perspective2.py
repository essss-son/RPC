import requests
import json
import time

 替换为你的 API 密钥
API_KEY = 'AIzaSyBYhRANm3GtxdNyLgxe3nZvFJ4pH5w6Oz0'
ENDPOINT = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze'

 输入的 TXT 文件路径
input_file = 'gpt2_medium_imdb_attr0.txt'
input_file = 'jigsaw_toxic_gpt2_all_prefix128_ck31260_length_64.txt'
 初始化统计变量
total_toxicity_score = 0
line_count = 0


 每分钟最大请求次数
MAX_REQUESTS_PER_MINUTE = 60

 设置每个请求之间的时间间隔（秒）
time_interval = 60 / MAX_REQUESTS_PER_MINUTE   1分钟 / 60次请求

 打开文件并逐行读取
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        text = line.strip()   去掉首尾空白字符
        if not text:   跳过空行
            continue

         设置请求的 JSON 数据
        data = {
            'comment': {'text': text},
            'languages': ['en'],   可以指定文本的语言
            'requestedAttributes': {'TOXICITY': {}}
        }

         发送请求
        response = requests.post(
            f'{ENDPOINT}?key={API_KEY}',
            data=json.dumps(data),
            headers={'Content-Type': 'application/json'}
        )

         解析和输出结果
        if response.status_code == 200:
            result = response.json()
            toxicity_score = result['attributeScores']['TOXICITY']['summaryScore']['value']
            print(f'Text: {text}')
            print(f'Toxicity score: {toxicity_score}\n')

             累加毒性分数并计数
            total_toxicity_score += toxicity_score
            line_count += 1
        else:
            print(f'Error: {response.status_code}')
            print(response.text)
            break   遇到错误时停止运行

         限制请求频率
        if line_count % MAX_REQUESTS_PER_MINUTE == 0:
            print("Reached the maximum requests per minute, waiting for the next minute...")
            time.sleep(60)   等待1分钟

        else:
            time.sleep(time_interval)   每次请求之间等待一定时间


 计算并输出平均毒性分数
if line_count > 0:
    average_toxicity_score = total_toxicity_score / line_count
    print(f'Average Toxicity Score: {average_toxicity_score:.4f}')
else:
    print('No valid text lines were processed.')
