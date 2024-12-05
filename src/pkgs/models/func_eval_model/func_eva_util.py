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
    return daily_diet_str if daily_diet_str else '无'

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
    img_time = []
    for info in diet_info:
        time_key = info.get('diet_time', '').split(' ')[-1].split(':')[0]
        if time_key:
            img_time.append(time_key)
    night_low = {}
    day_high = {}
    for key in buckets.keys():
        if int(key) in [int(i) - 1 for i in img_time]:
            res.append(buckets[key][len(buckets[key]) // 3])
            res.append(buckets[key][len(buckets[key]) // 3 * 2])
        if int(key) in [int(i) + 1 for i in img_time]:
            res.append(buckets[key][len(buckets[key]) // 3])
            res.append(buckets[key][len(buckets[key]) // 3 * 2])
        if int(key) in [int(i) + 2 for i in img_time]:
            res.append(buckets[key][len(buckets[key]) // 3])
            res.append(buckets[key][len(buckets[key]) // 3 * 2])
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
            for i in buckets[key]:
                if not day_high:
                    day_high = i
                elif day_high['value'] < i['value']:
                    day_high = i
    res.append(night_low)
    res.append(day_high)
    unique_tuples = set(tuple(sorted(d.items())) for d in res)
    res = [dict(t) for t in unique_tuples]
    res = sorted(res, key=lambda item: item.get('time', ''))
    return res


daily_diet_eval_prompt = """# 请你扮演一位经验丰富的营养师，对我提交的一日食物信息做出合理评价和建议。

## 个人信息：
{0}

## 当天饮食及每餐评价信息：
{1}

## 一日血糖信息：
{2}

## 管理场景：
{3}

## 输出要求
  - 输出可能包含的维度有：血糖稳定性评估、餐后血糖波动可能与饮食的关联并且评价食物选择是否合理、饮食待改善建议3个维度。
  - 每个维度的评价可以换行显示，可参考输出示例。
  - 血糖稳定性评估请从餐后血糖波动、最大血糖波动维度来分析，识别异常血糖信息，并给予提醒，比如对低血糖情况应给出建议。如果没有血糖数据，该评价维度可忽略。
  - 饮食待改善建议可以从用餐时间、用餐规律性、食物搭配等维度输出。
  - 若一日餐次饮食信息有缺失，可以给予提醒规律用餐的重要性。
  - 饮食待改善建议避免输出列表。
  - 输出内容要求简洁自然，通俗易懂，符合营养学观点。
  - 整体字数控制在300字以内。
 
## 输出示例
从你的血糖数据来看，餐后血糖波动较大，尤其是午餐后血糖显著升高至14mmol/L，之后虽有所下降，但仍高于理想范围。全天最大血糖波动也较大，从早餐后的12mmol/L到睡前的4mmol/L，波动幅度超过8mmol/L。需特别注意低血糖情况，应警惕夜间低血糖风险。
午餐蒸红薯是高升糖食物，且咸菜可能含较多盐分，可能是导致午餐后血糖升高的原因之一，不利于血糖控制。晚餐食物以蔬菜为主，血糖波动较小，但缺乏足够的蛋白质和主食，可能导致后续血糖过低。
建议调整午餐主食，选择低升糖指数的食物，如全麦面包或燕麦片，并增加绿叶蔬菜的摄入。晚餐应添加一份蛋白质来源，如鸡胸肉或豆腐，并搭配一份适量主食，如糙米饭或全麦面包，以确保营养均衡，同时避免血糖过低。整体饮食应控制总量，分餐次合理搭配，有助于稳定血糖。

Begins!"""


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