import requests
import json

 替换为你的 API 密钥
API_KEY = 'AIzaSyBYhRANm3GtxdNyLgxe3nZvFJ4pH5w6Oz0'
ENDPOINT = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze'

 定义待分析的文本
text = "I hate this so much! You're awful."

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
    print(f'Toxicity score: {toxicity_score}')
else:
    print(f'Error: {response.status_code}')
    print(response.text)
