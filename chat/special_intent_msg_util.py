from chat.constant import EXT_USRINFO_TRANSFER_INTENTCODE, default_prompt
from src.prompt.model_init import chat_qwen
from utils.Logger import logger

role_map = {
        '0': 'user',
        '1': 'user',
        '2': 'doctor',
        '3': 'assistant'
}

def get_userInfo_msg(prompt, history, intentCode):
    oo = chat_qwen(prompt, verbose=False,
            temperature=0.7, top_p=0.8, max_tokens=200)
    if '询问' in oo or '提问' in oo or '转移' in oo or '未知' in oo:
        his =[{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        his = ' '.join([h['role']+':'+h['content'] for h in his])
        query = default_prompt + his + ' user:'
        print('输入为：' + oo)
        oo = chat_qwen(query, verbose=False, temperature=0.7, top_p=0.8, max_tokens=200)
        intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
    
    return {'end':True, 'message':oo, 'intentCode':intentCode}


def get_reminder_tips(prompt, history, intentCode, model='Baichuan2-7B-Chat'):
    logger.debug('remind prompt: ' + prompt)
    model_output = chat_qwen(query=prompt, verbose=False, do_sample=False, 
            temperature=0.1, top_p=0.2, max_tokens=500, model=model)
    logger.debug('remind model output: ' + model_output)
    if model_output.startswith('（）'):
        mo = model_output[2:].strip()
    return {'end':True, 'message':mo, 'intentCode':intentCode}







