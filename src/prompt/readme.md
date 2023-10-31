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

## 现存问题
- 当前初始交互上下文已达到3000左右, 首token latency较长
- 长程记忆衰减，末端比较难严格遵循开头的提示
- 建议分治, 识别出具体日程任务, 针对性细化流程写sys prompt, 进行局部assistant user对话

## 较好的case

```html
output:  Thought: 需要询问用户开会的具体时间
call func:  ask_for_time
arguments:  {"task": "开会", "ask": "请问您希望会议在什么时间开始？"}
tool input: 晚上7点半
======================================================================
output:  Thought: 用户已经回答了开会的具体时间，可以创建日程了
call func:  create_schedule
arguments:  {"task": "开会", "time": "2023-10-31T19:30:00", "ask": "日程已创建成功，提醒时间为2023-10-31T19:30:00"}
tool input: 改到4点半
======================================================================
output:  Thought: 需要询问用户是否确认修改日程
Action: ask
Action Input: 请问您确认将开会时间修改为2023-10-31 16:42:28吗？
Thought: 用户确认修改日程
call func:  modify_schedule
arguments:  {"task": "开会", "time": "2023-10-31T16:42:28", "ask": "日程已修改成功"}
tool input: 算了，取消吧
======================================================================
output:  Thought: 用户取消了日程
call func:  cancel_schedule
arguments:  好的，日程已取消
请问您希望会议在什么时间开始？
```

## raw context

```html
<|im_start|>system
You are a powerful schedule management tool, please follow the following requirements to complete the corresponding tasks：
- Answer the following questions as best you can.
- You have access to the following APIs:

create_schedule: Call this tool to interact with the 创建日程 API. What is the 创建日程 API useful for? 一个用于创建日程的工具,如果日程时间不明确,先使用ask_for_time工具询问日程时间. Format the arguments as a JSON object. Parameters: [{"name": "task", "description": "日程名称", "required": true, "schema": {"type": "string"}}, {"name": "time", "description": "提醒的时间", "required": true, "schema": {"type": "string", "format": "timestamp"}}, {"name": "ask", "description": "告知用户日程已创建成功及提醒的时间", "required": true, "schema": {"type": "string"}}]

ask_for_time: Call this tool to interact with the 询问日程时间 API. What is the 询问日程时间 API useful for? 创建日程时向用户询问时间的工具. Format the arguments as a JSON object. Parameters: [{"name": "task", "description": "日程名称", "required": true, "schema": {"type": "string"}}, {"name": "ask", "description": "向用户询问日程提醒的时间", "required": true, "schema": {"type": "string"}}]

cancel_schedule: Call this tool to interact with the 取消日程 API. What is the 取消日程 API useful for? 取消日程是已给帮助用户取消当前日程的工具. Format the arguments as a JSON object. Parameters: [{"name": "task", "description": "日程名称", "required": true, "schema": {"type": "string"}}, {"name": "ask", "description": "告知用户日程已取消", "required": true, "schema": {"type": "string"}}]

modify_schedule: Call this tool to interact with the 修改日程 API. What is the 修改日程 API useful for? 修改日程是一个帮助用户修改当前日程的工具，可使用本工具修改对应日程时间. Format the arguments as a JSON object. Parameters: [{"name": "task", "description": "日程名称", "required": true, "schema": {"type": "string"}}, {"name": "time", "description": "当前用户希望日程提醒的时间", "required": true, "schema": {"type": "string", "format": "timestamp"}}, {"name": "ask", "descripton": "告知用户日程修改成功", "required": true, "schema": {"type": "string"}}]

query_schedule: Call this tool to interact with the 查询日程 API. What is the 查询日程 API useful for? 查询日程是一个帮助用户查询特定的日程信息的工具. Format the arguments as a JSON object. Parameters: [{"name": "task", "description": "日程名称", "required": true, "schema": {"type": "string"}}]

- Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [create_schedule, ask_for_time, cancel_schedule, modify_schedule, query_schedule]
Action Input: the input to the action, should be a JSON object
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)

当前日程:
[{'task': '开会', 'time': '2023-10-31T16:42:28', 'ask': '日程已修改成功'}]

Begin!<|im_end|>
<|im_start|>user
Question: 现在的时间是2023-10-31 16:42:28
下午开会,提前叫我<|im_end|>
<|im_start|>assistant
Thought: 需要询问用户开会的具体时间
Action: ask_for_time
Action Input: 晚上7点半
Thought: 用户已经回答了开会的具体时间，可以创建日程了
Action: create_schedule
Action Input: 改到4点半
Thought: 需要询问用户是否确认修改日程
Action: ask
Action Input: 请问您确认将开会时间修改为2023-10-31 16:42:28吗？
Thought: 用户确认修改日程
Action: modify_schedule
Action Input: 算了，取消吧<|im_end|>
<|im_start|>assistant
```

