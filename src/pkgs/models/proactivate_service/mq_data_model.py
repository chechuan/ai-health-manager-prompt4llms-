from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel

class Condition(BaseModel):
    mp: Optional[str] = None
    oper: Optional[str] = None
    vse: Optional[str] = None
    ma: Optional[str] = None
    es: Optional[str] = None

class Sequence(BaseModel):
    operator: str
    value: Optional[str] = None
    sequence_followed_by: Optional[str] = None
    window: Optional[str] = None
    outside_window: Optional[str] = None
    not_operator: Optional[bool] = None

class TaskBrief(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[str] = None

class SenseDefinition(BaseModel):
    condition: Condition
    sequences: List[Sequence]
    summary: str
    task_brief: TaskBrief

## mq input param
class Payload(BaseModel):
    level: str
    segment: str
    mask: str
    sense_definition: SenseDefinition
    operator_type: str


## task data model
class ExecutableTask(BaseModel):
    """
    Represents an executable task with all possible fields.
    Corresponds to the table structure in the image.
    """
    taskId: str
    tenantId: str
    type: str
    priority: Optional[Union[int, str]] = None
    scheduledTime: Optional[str] = None
    idempotentId: Optional[str] = None
    cancelTaskId: Optional[str] = None
    params: Optional[dict[str, object]] = None
    context: Optional[dict[str, object]] = None
    configId: Optional[str] = None
    traceId: Optional[str] = None
    version: Optional[Union[int, str]] = None

class TaskParams(BaseModel):
    """
    请求参数模型
    对应表格数据：
    | 参数名       | 必填 | 类型   | 描述                                                  |
    | ------------ | ---- | ------ | ----------------------------------------------------- |
    | taskType     | 是   | string | 任务类型                                              |
    | scheduleTime | 是   | string | 定时时间，如果立即执行传空                            |
    | tenantId     | 否   | string | 租户id                                                |
    | configId     | 否   | string | 配置id ，目前可以传空                                 |
    | traceId      | 否   | string | 用户的意图                                            |
    | contextData  | 否   | Object | 透传对象,用于回调时恢复上下文                         |
    | callBackData | 是   | Object | 回调数据内容                                          |
    """
    taskType: str
    scheduleTime: Optional[str] = None  # 或者可以用 datetime 类型，但需要处理空字符串的情况
    tenantId: Optional[str] = None
    configId: Optional[str] = "1"
    traceId: Optional[str] = None
    contextData: Optional[Dict[str, Any]] = None  # 或者可以定义更具体的模型
    callBackData: Optional[Dict[str, Any]] = None  # 或者可以定义更具体的模型


class TaskType():
    """
    任务类型枚举
    对应表格数据：
    | type                      | 描述                         | 参数                                                                 | 用途                                         |
    |---------------------------|------------------------------|----------------------------------------------------------------------|---------------------------------------------|
    | SEND_NOTIFICATION         | 通知推送（App/PC/短信/邮件） | channel (SM/SFMAIL/PUSH), message, title(可选)                       | 血糖异常、日程提醒、系统告警                 |
    | UPDATE_CALENDAR           | 日程系统更新                 | calendarId, startTime, endTime, title, description                  | 帮用户自动创建/修改日程                     |
    | CALL_API                  | 调用第三方或内部HTTP接口     | uri, method, headers, body                                          | 身份验证、数据同步、触发外部工作流           |
    | GENERATE_REPORT           | 报表/文档生成                | reportType, parameters                                              | 周报/月报、健康报告                         |
    | RUN_WORKFLOW              | 启动内部工作流/任务链        | workflowId, inputs                                                  | 复杂业务流程编排                            |
    | SEND_HEALTH_ALERT         | 专用健康预警任务             | threshold, LastMealTime, dietImageUrl                               | 针对血糖/心率等健康场景的专用通知            |
    | NO_DIET_NO_COMMENT_ALERT  | 健康场景——无饮食记录提示     | message                                                             | 用户未上传饮食图片时的提示                   |
    | REQUEST_DIET_IMAGE        | 请求用户上传饮食图片         | message, channel                                                   | 当检测到血糖异常但无记录时，请求用户上传     |
    """

    SEND_NOTIFICATION = "SEND_NOTIFICATION"
    UPDATE_CALENDAR = "UPDATE_CALENDAR"
    CALL_API = "CALL_API"
    GENERATE_REPORT = "GENERATE_REPORT"
    RUN_WORKFLOW = "RUN_WORKFLOW"
    SEND_HEALTH_ALERT = "SEND_HEALTH_ALERT"
    NO_DIET_NO_COMMENT_ALERT = "NO_DIET_NO_COMMENT_ALERT"
    REQUEST_DIET_IMAGE = "REQUEST_DIET_IMAGE"