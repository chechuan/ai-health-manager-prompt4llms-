import pandas as pd
import json
import re
import os

# 获取脚本文件所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

# config字典，集中管理所有可配置项
config = {
    'log_dir': os.path.join(root_dir, 'logs'),  # 日志文件夹路径
    'output_dir': os.path.join(root_dir, '.cache'),  # 输出文件夹路径
    'json_file': os.path.join(root_dir, '.cache', 'logs.json'),  # JSON格式的日志输出文件路径
}

# 读取日志文件夹下的所有日志文件
log_files = [os.path.join(config['log_dir'], f) for f in os.listdir(config['log_dir']) if
             f.endswith('.csv') or f.endswith('.csv.gz')]

# 合并所有日志文件内容
log_data = pd.DataFrame()
for file_path in log_files:
    if file_path.endswith('.csv.gz'):
        log_data = pd.concat([log_data, pd.read_csv(file_path)], ignore_index=True)
    elif file_path.endswith('.csv'):
        log_data = pd.concat([log_data, pd.read_csv(file_path)], ignore_index=True)

# 解析日志文件内容并生成JSON文件
json_file_path = config['json_file']
os.makedirs(config['output_dir'], exist_ok=True)

log_entries = []
for index, row in log_data.iterrows():
    content = row.get('content', '')
    if 'AIGC Functions' in content and 'LLM Output' in content:
        # 提取AIGC Functions后面的标题文本
        title_match = re.search(r'AIGC Functions\s*(.*?)\s*LLM Output', content)
        title = title_match.group(1).strip() if title_match else 'Unknown'

        # 提取LLM Output后面的内容
        llm_output_match = re.search(r'LLM Output:\s*(.*)', content, re.DOTALL)
        llm_output = llm_output_match.group(1).strip() if llm_output_match else ''

        log_entry = {
            "title": title,
            "llm_output": llm_output
        }
        log_entries.append(log_entry)

with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(log_entries, json_file, ensure_ascii=False, indent=4)

print(f"日志文件已成功解析并写入到 {json_file_path}")
