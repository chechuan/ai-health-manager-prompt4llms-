import yaml, os
from collections import OrderedDict

def get_func_eval_prompt(name):
    file_path = 'data/prompt_data/func_eval_prompt.yaml'
    dic = yaml.load(open(file_path, encoding="utf-8"), Loader=yaml.FullLoader)
    return dic.get(name, '')

def get_history_info(history):
    his_str = ''
    for i in history:
        his_str += f"{i['send_time']}: {i['role']}: {i['content']}\n"
    return his_str

def get_sch_str(schedule):
    sch_str = ''
    for s in schedule:
        sch_str += f"{s.get('time', '')} {s.get('name', '')}\n"
    return sch_str

def get_daily_blood_glucose_str(daily_blood_glucose):
    bg_str = ''
    for i in daily_blood_glucose:
        bg_str += f"测量时间：{i.get('time', '')}  测量值：{i.get('value', '')}\n"
    return bg_str

def get_daily_diet_str(daily_diet_info):
    daily_diet_str = ''
    for i in daily_diet_info:
        diet_info = ''
        for info in i.get('diet_info', []):
            if info:
                diet_info += f"{info.get('count', '')}{info.get('unit', '')}{info.get('foodname', '')}，"
        daily_diet_str += f"就餐时间：{i.get('diet_time', '')} 就餐食物：{diet_info}。医生当餐评价：{i.get('diet_eval', '无')}\n"
    return daily_diet_str

def get_daily_key_bg(bg_info, diet_info):
    res = []
    buckets = OrderedDict()
    for info in bg_info:
        time = info.get('time', '').split(' ')[-1]
        key = buckets.get(time.split(':')[0], '')
        if not key:
            buckets[time.split(':')[0]] = [info]
        else:
            buckets[time.split(':')[0]].append(info)
    img_time = OrderedDict()
    for info in diet_info:
        time_key = info.get('diet_time', '').split(' ')[-1].split(':')[0]
        if time_key:
            img_time[time_key] = time_key
    night_low = {}
    day_high = {}
    for key in buckets.keys():
        if int(key) < 6 or int(key) > 21:
            for i in buckets[key]:
                if not night_low:
                    night_low = i
                elif night_low['value'] > i['value']:
                    night_low = i
            if int(key) % 2 == 0:
                res.append(buckets[key][0])
        elif 5 < int(key) < 22:
            res.append(buckets[key][0])
            if not day_high:
                day_high = i
            elif day_high['value'] < i['value']:
                day_high = i
    res.append(night_low)
    res.append(day_high)
    unique_tuples = set(tuple(sorted(d.items())) for d in res)
    res = [dict(t) for t in unique_tuples]
    res = sorted(res, key=lambda item: item["time"])
    return res









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