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
CUDA_VISIBLE_DEVICES=4 python3 -m fastchat.serve.vllm_worker --port 20003 --worker-address http://localhost:20003 --model-names Qwen1.5-14B-Chat --model-path /home/songhaoyang/.cache/modelscope/hub/qwen/Qwen1.5-14B-Chat/ --controller-address http://localhost:20001 --gpu-memory-utilization 0.625
```
## Tips
- 当前使用vllm backend engine部署模型,最大长度受此函数控制,配置文件中不含`rope_scaling`参数时会默认取2048
- 可以直接修改模型配置文件中的`SEQUENCE_LENGTH_KEYS`参数使可推理总tokens扩展
```python
SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]
def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = config.rope_scaling["factor"]
    else:
        rope_scaling_factor = 1

    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048
```
