# 所需资源
1. 显卡
2. 环境
3. 模型权重
4. FastChat代码库

# 部署流程
## step1 登录A800服务器 切目录、环境
```sh
cd /data/songhaoyang/FastChat && conda activate shy_dev
```
## 启动controller
```
python3 -m fastchat.serve.controller --port 20001
```
## 启动api server
```sh
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 26925 --controller-address http://localhost:20001
```
## 启动model worker
```sh
CUDA_VISIBLE_DEVICES=4 python3 -m fastchat.serve.model_worker --port 20002 --worker-address http://localhost:20002 --model-names Baichuan2-7B-Chat --model-path /home/songhaoyang/.cache/modelscope/hub/baichuan-inc/Baichuan2-7B-Chat/ --controller-address http://localhost:20001
CUDA_VISIBLE_DEVICES=4 python3 -m fastchat.serve.vllm_worker --port 20003 --worker-address http://localhost:20003 --model-names Qwen-14B-Chat --model-path /home/songhaoyang/.cache/modelscope/hub/qwen/Qwen-14B-Chat/ --controller-address http://localhost:20001 --gpu-memory-utilization 0.625
```