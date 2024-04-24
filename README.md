# ChangeLog

-   2024 年 1 月 16 日 18:15:05 调整`model_config` && 开发 SearchQAChain
-   2024 年 1 月 15 日 10:25:45 add llm_token && 取消日程优化阶段性 && 年夜饭共策针对实际问题优化
-   2023 年 11 月 13 日 14:32:47 流程中使用到的 prompt,切到从 mysql 读取,添加`/reload_prompt`方法
-   2023 年 11 月 8 日 10:01:44 prompt 系列工程接入数字人主逻辑
-   2023 年 11 月 2 日 11:07:08 输出模式适配
-   2023 年 10 月 31 日 16:54:36 日程管理模块优化, 对应 readme 补充
-   2023 年 10 月 27 日 14:48:46 add 日程操作模块
-   2023 年 10 月 27 日 10:19:15 补充知识工程 prompt readme
-   2023 年 10 月 26 日 18:07:49 update src.prompt.factory 引入知识工程
-   2023 年 10 月 23 日 13:58:02 update langchain default model_name
-   2023 年 10 月 20 日 14:52:48 update funcCall
-   2023 年 10 月 19 日 16:51:39 更新 dev plan

# Deploy

见`doc/deploy_proj`

## 注意事项

1. 新增事件的 mysql 同步,对应环境的配置文件更新
2. 新增依赖维护 requirements.txt
3. dev -> test -> prod
4. fat(目前为展厅环境, 古早版本,可以考虑展厅版本单独开几个分支开发)
5. 新增模型集成在http://10.228.67.99:26928/
6. 部署前信息同步

# Plan

## v0.1

-   [x] 对话流程, prompt 组装跑通
-   [x] `REACT_INSTRUCTION`中明确要求首次应当调用获取流程`tool`获取处理流程
-   [x] 提供获取处理流程工具,处理流程中应包含当前支持的类别,工作流中第一次让 LLM 选择处理流程填充到`REACT_INSTRUCTION`中

## v0.2

-   [x] 开发 langchain、graph 知识查询工具

## v0.3

-   [x] 先验知识模块开发
-   [x] 日程管理模块开发

## v0.4

-   [x] 提示模板、代码拆分

## v0.5

-   [x] 角色事件知识接入对话流程

## v0.6

-   [ ] 通用 ai-naive 事件处理流程
