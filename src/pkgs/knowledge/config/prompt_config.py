from langchain.prompts.prompt import PromptTemplate

# prompt模板使用Jinja2语法，简单点就是用双大括号代替f-string的单大括号
# 本配置文件支持热加载，修改prompt模板后无需重启服务。


# LLM对话支持的变量：
#   - input: 用户输入内容

# 知识库和搜索引擎对话支持的变量：
#   - context: 从检索结果拼接的知识文本
#   - question: 用户提出的问题

# Agent对话支持的变量：

#   - tools: 可用的工具列表
#   - tool_names: 可用的工具名称列表
#   - history: 用户和Agent的对话历史
#   - input: 用户输入内容
#   - agent_scratchpad: Agent的思维记录

PROMPT_TEMPLATES = {
    "completion": {
        "default": "{input}"
    },

    "llm_chat": {
        "default": "{{ input }}",

        "py":
            """
        你是一个聪明的代码助手，请你给我写出简单的py代码。 \n
        {{ input }}
        """
        ,
    },

    "knowledge_base_chat": {
        "default":(
            """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>""",
            """<已知信息>{{ context }}</已知信息>""",
            """<问题>{{ question }}</问题>"""),
        "text": (
            """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，答案请使用中文。 </指令>""",
            """<已知信息>{{ context }}</已知信息>""",
            """<问题>{{ question }}</问题>"""
            ),
        "Empty":  # 搜不到内容的时候调用，此时没有已知信息，这个Empty可以更改，但不能删除，会影响程序使用
            (
                """<指令>请根据用户的问题，进行简洁明了的回答</指令>""",
                """<问题>{{ question }}</问题>"""
            )
    },

    "search_engine_chat": {
        "default":(
            "<指令>这是我搜索到的互联网信息，请你根据这些信息进行提取并有调理，简洁的回答问题。如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。 </指令>\n"
            "<已知信息>{{ context }}</已知信息>\n"
            "<问题>{{ question }}</问题>\n"
        ),
        "search":(
            "<已知信息>\n{{ context }}\n</已知信息>\n\n"
            "请你根据以上提供的信息，简洁和专业的来回答问题。如果无法从中得到答案，请说: 根据已知信息无法回答该问题, 答案请使用中文。"
        ),
        "Empty": "请根据用户的问题，进行简洁明了的回答"  # 搜不到内容的时候调用，此时没有已知信息，这个Empty可以更改，但不能删除，会影响程序使用
    },

    "agent_chat": {
        "default":
            """
        Answer the following questions as best you can. If it is in order, you can use some tools appropriately.You have access to the following tools:

        {tools}

        Please note that the "知识库查询工具" is information about the "西交利物浦大学" ,and if a question is asked about it, you must answer with the knowledge base，
        Please note that the "天气查询工具" can only be used once since Question begin.

        Use the following format:
        Question: the input question you must answer1
        Thought: you should always think about what to do and what tools to use.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question


        Begin!
        history:
        {history}
        Question: {input}
        Thought: {agent_scratchpad}
        """,

        "AgentLM":
            """
        <SYS>>\n
        You are a helpful, respectful and honest assistant.
        </SYS>>\n
        Answer the following questions as best you can. If it is in order, you can use some tools appropriately.You have access to the following tools:

        {tools}.

        Use the following steps and think step by step!:
        Question: the input question you must answer1
        Thought: you should always think about what to do and what tools to use.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin! let's think step by step!
        history:
        {history}
        Question: {input}
        Thought: {agent_scratchpad}

        """,

        "中文版本":
            """
        你的知识不一定正确，所以你一定要用提供的工具来思考，并给出用户答案。
        你有以下工具可以使用:
        {tools}

        请请严格按照提供的思维方式来思考，所有的关键词都要输出，例如Action，Action Input，Observation等
        ```
        Question: 用户的提问或者观察到的信息，
        Thought: 你应该思考该做什么，是根据工具的结果来回答问题，还是决定使用什么工具。
        Action: 需要使用的工具，应该是在[{tool_names}]中的一个。
        Action Input: 传入工具的内容
        Observation: 工具给出的答案（不是你生成的）
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: 通过工具给出的答案，你是否能回答Question。
        Final Answer是你的答案

        现在，我们开始！
        你和用户的历史记录:
        History:
        {history}

        用户开始以提问：
        Question: {input}
        Thought: {agent_scratchpad}
        """,
    },
}

_DEFAULT_SEARCH_QA_TEMPLATE = """请你根据我检索到的信息, 基于其中的知识, 有条理、简洁的回答问题。如果无法从中得到答案，请说:未搜索到相关内容。

# 检索信息
{context}

# 问题
{question}

# 回答
"""
SEARCH_QA_PROMPT = PromptTemplate(
    template=_DEFAULT_SEARCH_QA_TEMPLATE, input_variables=["context", "question"]
)
_DEFAULT_SEARCH_HISTORY_TEMPLATE = """<指令>这是我搜索到的互联网信息，请你根据这些信息进行提取并有条理的回答问题。如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。 </指令>
<已知信息>\n{context}\n</已知信息>
<历史会话>\n{history}\n</历史会话>
<问题>\n{question}\n</问题>"""
SEARCH_QA_HISTORY_PROMPT = PromptTemplate(
    template=_DEFAULT_SEARCH_HISTORY_TEMPLATE, input_variables=["context", "history", "question"]
)