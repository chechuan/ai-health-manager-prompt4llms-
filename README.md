# ChangeLog
- 2023年11月8日10:01:44 prompt系列工程接入数字人主逻辑
- 2023年11月2日11:07:08 输出模式适配
- 2023年10月31日16:54:36 日程管理模块优化, 对应readme补充
- 2023年10月27日14:48:46 add 日程操作模块
- 2023年10月27日10:19:15 补充知识工程prompt readme
- 2023年10月26日18:07:49 update src.prompt.factory 引入知识工程
- 2023年10月23日13:58:02 update langchain default model_name
- 2023年10月20日14:52:48 update funcCall
- 2023年10月19日16:51:39 更新dev plan

# 交互流程

# Plan
## v0.1
- [x] 对话流程, prompt组装跑通
- [x] `REACT_INSTRUCTION`中明确要求首次应当调用获取流程`tool`获取处理流程
- [x] 提供获取处理流程工具,处理流程中应包含当前支持的类别,工作流中第一次让LLM选择处理流程填充到`REACT_INSTRUCTION`中
## v0.2
- [x] 开发langchain、graph 知识查询工具
## v0.3
- [x] 先验知识模块开发
- [x] 日程管理模块开发
## v0.4