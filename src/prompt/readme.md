# 先验知识体系引入
## 模块说明
1. 确定知识类型及内容拼接顺序
2. 确定各类型对应的TEMPLATE
3. `factory.py`中`promptEngine._call`定义了组装流程及拼接关键字


## 使用说明
```python
from src.prompt.factory import baseVarsForPromptEngine
# 1. 确定初始变量
args = baseVarsForPromptEngine()
# 2. call promptEngine
pe = promptEngine()
prompt = pe._call(args, concat_keyword=",")
```