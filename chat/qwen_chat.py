from chat.qwen_react_util import *
from chat.qwen_tool_config import *

role_map = {
        '0': '<用户>:',
        '1': '<用户>:',
        '2': '<医生>:',
        '3': '<智能健康管家>:'
}

cls_prompt = '你作为智能健康管家，需要根据用户的对话内容，判断需要从工具列表中调用哪个工具完成用户的需求，工具列表为：调用外部知识库、进一步询问用户的情况，直接回复用户问题。对话历史为：'

class Chat(object):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def get_tool_name(self, text):
        if '外部知识' in text:
            return '调用外部知识库'
        elif '询问用户' in text:
            return '进一步询问用户的情况'
        elif '直接回复' in text:
            return '直接回复用户问题'
        else:
            return '直接回复用户问题'

    def generate(self, input_text):
        ids = self.tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).to(self.model.device) 
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=400,
            do_sample=False,
            temperature=0
        )
        out_text = self.tokenizer.decode(out[0])
        return out_text.split('请输出需调用的工具：')[-1]


    def get_qwen_history(self, history):
        hs = []
        hcnt = {}
        for cnt in history:
            if len(hcnt.keys()) == 2:
                hs.append(hcnt)
                hcnt = {}
            if cnt['role'] == '1' or cnt['role'] == '0':
                if 'user' in hcnt.keys():
                    hcnt['bot'] = ''
                    hs.append(hcnt)
                    hcnt = {}
                hcnt['user'] = cnt['content']
            else:
                if 'bot' in hcnt.keys():
                    hcnt['user'] = ''
                    hs.append(hcnt)
                    hcnt = {}
                hcnt['bot'] = cnt['content']
        if hcnt:
            if 'user' in hcnt.keys():
                hcnt['bot'] = ''
            else:
                hcnt['user'] = ''
            hs.append(hcnt)
            hcnt = {}
        hs = [(x['user'], x['bot']) for x in hs]
        return hs



    def run_prediction(self, history, prompt, intentCode):
        hist = [{'role': role_map.get(str(cnt['role']), '1'), 'content':cnt['content']} for cnt in history]
        hist = [cnt['role'] + cnt['content'] for cnt in hist]
        his = cls_prompt + ' '.join(hist) + '\n请输出需调用的工具：'
        tool = self.get_tool_name(self.generate(his))
        print('工具类型：' + tool)
        if tool == '进一步询问用户的情况' or tool == '直接回复用户问题':
            h = self.get_qwen_history(history)
            #history = build_input_text(h, [])
            response, qw_his = llm_with_plugin(history=h,
                    list_of_plugin_info=qwen_tools, model=self.model,
                    tokenizer=self.tokenizer)
        #elif tool == '调用外部知识库':


