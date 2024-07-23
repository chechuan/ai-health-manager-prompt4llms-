# -*- encoding: utf-8 -*-
'''
@Time    :   2024-07-23 15:55:43
@desc    :   提取日志文件并生成统计数据
@Author  :   车川
@Contact :   chechuan1204@gmail.com
'''

import os
import gzip
import json
import re
import sys
import pandas as pd
from collections import defaultdict
from loguru import logger

# 获取 extract_logs.py 文件所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

# config字典，集中管理所有可配置项
config = {
    'log_dir': os.path.join(root_dir, 'logs'),  # 日志文件夹路径
    'output_dir': os.path.join(root_dir, '.cache'),  # 输出文件夹路径
    'jsonlines_file': os.path.join(root_dir, '.cache', 'logs.jsonl'),  # jsonlines格式的日志输出文件路径
    'stats_file': os.path.join(root_dir, '.cache', 'stats.json')  # 统计数据输出文件路径
}

# 配置日志格式
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# 添加配置到logger
logger.remove()  # 移除默认的日志配置
logger.add(sys.stdout, format=LOG_FORMAT, level="DEBUG")

# 处理Unicode转义字符
def decode_unicode_escape(input_str):
    """
    将字符串中的Unicode转义字符解码
    """
    return input_str.encode('utf-8', errors='ignore').decode('unicode_escape')

# 清理并修正JSON字符串
def clean_json_string(json_str):
    """
    清理并修正JSON字符串
    """
    # 替换转义的引号
    json_str = json_str.replace('\\"', '"')
    # 替换换行符
    json_str = json_str.replace('\\n', ' ')
    return json_str

# 解析日志行
def parse_log_line(line):
    """
    解析日志行，提取Endpoint和Input Param字段
    """
    input_param_pattern = r'Input Param: ({.*})'  # 匹配Input Param的正则表达式
    endpoint_pattern = r'Endpoint: (.*?), Input Param: ({.*})'  # 匹配Endpoint和Input Param的正则表达式

    input_param_match = re.search(input_param_pattern, line)
    endpoint_match = re.search(endpoint_pattern, line)

    try:
        if endpoint_match:
            # 提取Endpoint和Input Param
            endpoint = endpoint_match.group(1)
            input_param_str = endpoint_match.group(2)
            input_param_str = clean_json_string(input_param_str)
            input_param = json.loads(input_param_str)
            return endpoint, input_param
        elif input_param_match:
            # 只提取Input Param
            input_param_str = input_param_match.group(1)
            input_param_str = clean_json_string(input_param_str)
            input_param = json.loads(input_param_str)
            return None, input_param
    except json.JSONDecodeError as e:
        pass
        # logger.error(f"JSONDecodeError: {e} in line: {line}")
        # logger.error(f"Problematic line: {line}")
    return None, None

# 提取日志
def extract_logs(log_dir):
    """
    从日志文件中提取日志数据
    """
    logs = []
    log_file_pattern = re.compile(r'.*\.log(\.gz)?|.*\.csv(\.gz)?')  # 匹配日志文件的正则表达式
    for root, _, files in os.walk(log_dir):
        for file in files:
            if log_file_pattern.match(file):
                file_path = os.path.join(root, file)
                open_func = gzip.open if file.endswith('.gz') else open  # 根据文件类型选择打开方式
                with open_func(file_path, 'rt', encoding='utf-8') as f:
                    has_data = False
                    if file.endswith('.csv') or file.endswith('.csv.gz'):
                        # 处理CSV文件
                        df = pd.read_csv(f)
                        for _, row in df.iterrows():
                            content = row.get('content', '')
                            if isinstance(content, str) and content:
                                endpoint, input_param = parse_log_line(content)
                                if input_param:
                                    log_entry = {"endpoint": endpoint, "input_param": input_param}
                                    logs.append(log_entry)
                                    has_data = True
                    else:
                        # 处理普通日志文件
                        for line in f:
                            if "Input Param" in line or "Endpoint" in line:
                                logger.debug(f"Processing line: {line.strip()}")
                                endpoint, input_param = parse_log_line(line)
                                if input_param:
                                    log_entry = {"endpoint": endpoint, "input_param": input_param}
                                    logs.append(log_entry)
                                    has_data = True
                                else:
                                    pass
                                    # logger.warning(f"Unable to parse line: {line.strip()}")
                    if not has_data:
                        pass
                        # logger.warning(f"No data found in file: {file_path}")
    return logs

# 保存数据到jsonlines格式文件
def save_jsonlines(data, output_file):
    """
    将数据保存为jsonlines格式文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

# 生成统计数据
def generate_statistics(logs):
    """
    根据日志数据生成统计数据
    """
    stats = defaultdict(int)
    for log in logs:
        intent_code = log['input_param'].get('intentCode', '')
        if intent_code.startswith('aigc_functions'):
            stats[intent_code] += 1
    return stats

# 主函数
def main():
    """
    主函数，执行日志提取和统计生成过程
    """
    os.makedirs(config['output_dir'], exist_ok=True)  # 确保输出目录存在

    logs = extract_logs(config['log_dir'])  # 提取日志数据

    save_jsonlines(logs, config['jsonlines_file'])  # 保存日志数据

    stats = generate_statistics(logs)  # 生成统计数据
    with open(config['stats_file'], 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()