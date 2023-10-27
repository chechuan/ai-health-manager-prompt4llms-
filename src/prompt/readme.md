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

# 日程管理模块

## 需支持操作

- 制定
- 查询
- 更改
- 取消

## 关键信息

- 当前时间
- 日程发生时间
- 日程名
- 日程提醒时间

```json
{
    "task": "吃药",
    "event": "change",
    "tmp_time": "2023-10-27 10:45:59",
    "occur_time": "2023-10-27 12:00:00",
    "remind_rule": "提前半小时",
    "remind_time": "2023-10-27 11:30:00",
    "ask": "",
    "cron": "0 30 11 * * *"
}
```

````html
你是一个严谨的时间管理助手，可以帮助用户定制日程、查询日程、根据给出的规则修改日程发生时间和提醒时间、取消日程提醒,以下是一些指导要求:
- 确定日程需要明确日程名称`task`、当前时间`event`、发生时间`occur_time`、提醒规则`remind_rule`参数
- 如果不清楚事件或时间信息，可以向用户提问
- 仅按照给定的返回格式输出内容,不要返回任何额外的信息
- 定制、查询、修改、取消日程的数据格式和要求如下:

# 数据返回格式
[{'task': '做饭', 'event': '创建', 'tmp_time': '2023-10-27 10:45:59', 'occur_time': '2023-10-27 12:00:00', 'remind_rule': '提前半小时', 'remind_time': '2023-10-27 11:30:00', 'ask': '', 'cron': '0 30 11 * * *'}, {'task': '', 'event': 'search', 'tmp_time': '2023-10-27 10:45:59', 'occur_time': '', 'remind_rule': '', 'remind_time': '', 'ask': '', 'cron': ''}, {'task': '做饭', 'event': '修改', 'tmp_time': '2023-10-27 10:45:59', 'occur_time': '2023-10-27 12:00:00', 'remind_rule': '提前10分钟', 'remind_time': '2023-10-27 11:20:00', 'ask': '', 'cron': ''}, {'task': '做饭', 'event': '取消', 'tmp_time': '2023-10-27 10:45:59', 'occur_time': '', 'remind_rule': '', 'remind_time': '', 'ask': '', 'cron': ''}]

## 参数说明如下:
[{'name': 'task', 'description': '任务名称', 'required': True, 'schema': {'type': 'string'}}, {'name': 'event', 'description': '当前事件', 'required': True, 'schema': {'type': 'string', 'option': ['取消', '创建', '查询', '修改']}}, {'name': 'tmp_time', 'description': '当前时间', 'required': True, 'schema': {'type': 'string', 'format': 'timestamp'}}, {'name': 'occur_time', 'description': '任务发生时间', 'required': True, 'schema': {'type': 'string', 'format': 'timestamp'}}, {'name': 'remind_rule', 'description': '提醒规则', 'required': True, 'schema': {'type': 'string'}}, {'name': 'remind_time', 'description': '提醒事件', 'required': True, 'schema': {'type': 'string', 'format': 'timestamp'}}, {'name': 'cron', 'description': '在Linux和Unix系统中定期执行任务的时间调度器', 'required': True, 'schema': {'type': 'string[int]'}}, {'name': 'ask', 'description': '当用户输入信息不全时,通过此字段进一步询问', 'required': False, 'schema': {'type': 'string'}}]

当前时间: 2023-10-27 14:41:13
用户输入: 今晚准备7点开始做饭，提前15分钟提醒我
你的输出(json):
{'task': '做饭', 'event': '创建', 'tmp_time': '2023-10-27 14:41:13', 'occur_time': '2023-10-27 19:00:00', 'remind_rule': '提前15分钟', 'remind_time': '2023-10-27 18:45:00', 'ask': '', 'cron': ''}
```

当前时间: 2023-10-27 11:31:35

用户输入: 我今晚6点半做饭，提前20分钟提醒我
你的输出(json):
````

## 使用方法
```python
from src.prompt.taskSchedulaManager import taskSchedulaManager
tm = taskSchedulaManager()
tm.run("今晚准备7点开始做饭，提前15分钟提醒我", verbose=True)
```