qwen_tools = [
        # {
        #     "name_for_human": "谷歌搜索",
        #     "name_for_model": "google_search",
        #     "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。"
        #     + " Format the arguments as a JSON object.",
        #     "parameters": [
        #         {
        #             "name": "search_query",
        #             "description": "搜索关键词或短语",
        #             "required": True,
        #             "schema": {"type": "string"},
        #         }
        #     ],
        # },
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
                    "description": "要询问患者的问题",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "直接回复用户",
            "name_for_model": "query_answer",
            "description_for_model":"与用户日常交流，如果用户的问题可直接回答，或回复用户的问候，使用此工具直接作答。" + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "answer",
                    "description": "用户问题的答案",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
    ]

