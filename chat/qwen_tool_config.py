qwen_tools = [
         #{
         #    "name_for_human": "搜索引擎",
         #    "name_for_model": "search",
         #    "description_for_model": "使用通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。" + " Format the arguments as a JSON object.",
         #    "parameters": [
         #        {
         #            "name": "search_query",
         #            "description": "搜索关键词或短语",
         #            "required": True,
         #            "schema": {"type": "string"},
         #        }
         #    ],
         #},
        #{
        #    "name_for_human": "文生图",
        #    "name_for_model": "image_gen",
        #    "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。"
        #    + " Format the arguments as a JSON object.",
        #    "parameters": [
        #        {
        #            "name": "prompt",
        #            "description": "英文关键词，描述了希望图像具有什么内容",
        #            "required": True,
        #            "schema": {"type": "string"},
        #        }
        #    ],
        #},
        #{
         #   "name_for_human": "夸克搜索",
         #   "name_for_model": "kuake_search",
         #   "description_for_model": "夸克搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。" + " Format the arguments as a JSON object.",
        #    "parameters": [
        #        {
        #            "name": "search_query",
        #            "description": "搜索关键词或短语",
        #            "required": True,
        #            "schema": {"type": "string"},
        #        }
        #    ],
        #},
        # {
        #     "name_for_human": "询问医生",
        #     "name_for_model": "ask_doctor_for_help",
        #     "description_for_model": "搜索工具不足以准确获取相关信息时，直接向专业医生寻求帮助，获取更多医学相关专业知识。" + " Format the arguments as a JSON object.",
        #     "parameters": [
        #         {
        #             "name": "question",
        #             "description": "要问的问题",
        #             "required": True,
        #             "schema": {"type": "string"},
        #         }
        #     ],
        # }, 
        {
            "name_for_human": "与用户交流",
            "name_for_model": "chat_with_user",
            "description_for_model": "与用户日常交流，当前返回的内容需要用户回答，或者用户提供的信息不足，使用此工具可以询问更多信息。" + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "question",
                    "description": "要询问用户的问题",
                    "name": "query",
                    "description": "要询问患者的问题",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        # {
        #     "name_for_human": "直接回复用户",
        #     "name_for_model": "query_answer",
        #     "description_for_model":"与用户日常交流，如果用户的问题可直接回答，或回复用户的问候，使用此工具直接作答。" + " Format the arguments as a JSON object.",
        #     "parameters": [
        #         {
        #             "name": "answer",
        #             "description": "用户问题的答案",
        #             "required": True,
        #             "schema": {"type": "string"},
        #         }
        #     ],
        # },
        #{
        #    "name_for_human": "获取处理流程",
        #    "name_for_model": "get_plan",
        #    "description_for_model": "使用此工具可获取不同问题对应的处理流程" + " Format the arguments as a JSON object.",
        #    "parameters": [
        #        {
        #            "name": "query",
        #            "description": "参数值为问题的类别，可选类别如下:'辅助诊断','日常对话','疾病/症状/体征异常问诊'",
        #            "required": True,
        #            "schema": {"type": "string"},
        #        }
        #    ],
        #},
        #{
        #    "name_for_human": "获取图谱知识",
        #    "name_for_model": "llm_with_graph",
        #    "description_for_model": "可以帮助你查询知识图谱中的信息，并返回相应的结果。这个工具能够理解你的查询意图，并从知识图谱中检索出最相关的知识。" + " Format the arguments as a JSON object.",
        #    "parameters": [
        #        {
        #            "name": "query",
        #            "description": "未知、困惑的问题，本工具将会针对此问题提供相关知识支持",
        #            "required": True,
        #            "schema": {"type": "string"},
        #        }
        #    ],
        #},
        {
            "name_for_human": "文档知识查询",
            "name_for_model": "llm_with_documents",
            "description_for_model": "文档知识查询可以从目标文档中查询问题相关专业知识, 专业程度/可信度更高" + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "query",
                    "description": "未知、困惑的问题，本工具将会针对此问题提供相关知识支持",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "搜索引擎",
            "name_for_model": "llm_with_search_engine",
            "description_for_model": "duckduckgo是一个功能强大通用搜索引擎,可访问互联网、查询百科知识、了解时事新闻等,其他工具无法检索问题相关知识时,可以使用搜索引擎" + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "query",
                    "description": "搜索关键词或短语",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        }
    ]

