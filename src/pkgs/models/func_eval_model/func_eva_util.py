import yaml, os

def get_func_eval_prompt(name):
    file_path = 'data/prompt_data/func_eval_prompt.yaml'
    dic = yaml.load(open(file_path, encoding="utf-8"), Loader=yaml.FullLoader)
    return dic.get(name, '')

def get_history_info(history):
    his_str = ''
    for i in history:
        his_str += f"{i['send_time']}: {i['role']}: {i['content']}\n"
    return his_str

def get_standard_img_type(text):
    if '运动' in text:
        return 'sport_image'
    elif '体重' in text:
        return 'weight_image'
    elif '饮食' in text:
        return 'diet_image'
    elif '其他' in text:
        return 'other_image'
    else:
        return 'other_image'


schedule_modify_prompt = """# 你是一名健康饮食管理助手，你需要根据`用户日程模版信息`和`用户对话交互信息`，调整用户的日程提醒信息。

  ## 用户日程模版信息：
  {0}

  ## 用户对话交互信息：
  {1}

  ## 更新日程模版信息时间：
  {2}
  
  ## 输出内容要求：
  - 主题要求为根据`用户对话交互信息`来更新用户日程模版信息。
  - 重点对比`用户对话交互信息`中对话的发送时间点和‘更新日程模版信息时间’，来识别日程模版信息中是否有已经发生或完成的事项。
  - 如果用户已完成了某个事项，可在日程模版信息中简短评价用户的该事项。
  - 如果`用户日程模版信息`中有的事项没有在`用户对话交互信息`中完成，可原样返回该事项提示内容。
  - 对于`用户日程模版信息`中的知识科普和宣教类内容事项，可原样输出，不做评论。
  - 对于`用户日程模版信息`中打卡类事项，如果`对话交互信息`中用户已完成本次的打卡的内容，则打卡事项内容处提示用户已完成打卡。
  - 不要更改日程模版的格式，按原模版格式输出。
  - 字数在200字以内
  
  Begins!"""