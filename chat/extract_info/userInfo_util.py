from src.prompt.model_init import chat_qwen
from chat.constant import default_prompt, EXT_USRINFO_TRANSFER_INTENTCODE

def get_userInfo_msg(promt, history, intentCode):
    model_output = chat_qwen(prompt, verbose=kwargs.get("verbose", False),
            temperature=0.7, top_p=0.8, max_tokens=200)
    if '询问' in oo or '提问' in oo or '转移' in oo or '未知' in model_output:
        model_output = chat_qwen(default_prompt, ['role':history[-1], 'content':history[-1]['content']], verbose=kwargs.get("verbose", False), temperature=0.7, top_p=0.8, max_tokens=200)
        intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
    
    return {'end':True, 'message':model_output, 'intentCode':intentCode}





