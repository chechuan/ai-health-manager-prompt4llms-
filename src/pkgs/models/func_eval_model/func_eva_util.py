import yaml, os
from collections import OrderedDict
from scipy.signal import find_peaks
from src.pkgs.models.func_eval_model.daily_image_analysis import get_meal_increase_glucose_periods
from datetime import datetime

# sport_category_mapping = {
#     '血糖异常提示': 'HS001',
#     '身体不适': 'HS002',
#     '急性疾病': 'HS003',
#     '慢性病急性发作': 'HS004',
#     '检查异常': 'HS005',
#     '低血糖症状': 'HS006',
#     '糖尿病并发症': 'HS007',
#     '环境因素': 'HS008',
#     '生活习惯': 'HS009',
#     '不适症状': 'HS010',
#     '女性特殊时期': 'HS011',
#     '突发事件': 'HS012'
# }

sport_category_mapping = {
    '血糖异常提示': 'HS001',
    '血压异常': 'HS002',
    '存在糖尿病并发症': 'HS003',
    '身体不适': 'HS004',
    '检查异常': 'HS005',
    '环境因素': 'HS006',
    '生活方式变化': 'HS007',
    '女性特殊时期': 'HS008',
    '突发事件': 'HS009'
}

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
        if not i:
            continue
        bg_str += f"测量时间：{i.get('time', '')}  测量值：{i.get('value', '')}\n"
    return bg_str

def get_daily_diet_str(daily_diet_info, daily_blood_glucose):
    daily_diet_str = ''
    # periods_str = get_meal_increase_glucose_periods(daily_diet_info, daily_blood_glucose)
    for idx, i in enumerate(daily_diet_info):
        diet_info = ''
        for info in i.get('diet_info', []):
            if info:
                diet_info += f"{info.get('count', '')}{info.get('unit', '')}{info.get('foodname', '')}，"
        daily_diet_str += f"就餐时间：{i.get('diet_time', '')} 就餐食物：{diet_info}。医生当餐评价：{i.get('diet_eval', '无')}\n"
        # if len(periods_str) == len(daily_diet_info):
        #     daily_diet_str += f"就餐时间：{i.get('diet_time', '')} 就餐食物：{diet_info}。医生当餐评价：{i.get('diet_eval', '无')}。 {periods_str[idx]}\n"
        # else:
        #     daily_diet_str += f"就餐时间：{i.get('diet_time', '')} 就餐食物：{diet_info}。医生当餐评价：{i.get('diet_eval', '无')}\n"

    return daily_diet_str if daily_diet_str else '无'

def get_meal_name(time_str):
    meal_time = datetime.strptime(time_str, '%H:%M:%S')
    if meal_time <= datetime.strptime('10:30:00', '%H:%M:%S'):
        return '早餐'
    elif datetime.strptime('14:30:00', '%H:%M:%S') > meal_time > datetime.strptime('10:30:00', '%H:%M:%S'):
        return '午餐'
    elif datetime.strptime('17:30:00', '%H:%M:%S') < meal_time < datetime.strptime('22:00:00', '%H:%M:%S'):
        return '晚餐'
    else:
        return '加餐'

def get_daily_key_bg(bg_info, diet_info):
    res = []
    buckets = OrderedDict()
    if not bg_info:
        bg_info = []
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
            img_time.append(info.get('diet_time', '').split(' ')[-1])
    night_low = {}
    day_high = {}
    peaks = []
    troughs = []
    meals_minmax_bg = {}
    for i in img_time:
        after_meal_data = []
        for key in buckets.keys():
            if int(key) == (int(i.split(':')[0]) - 1):
                meal_trough_idxs, _ = find_peaks([-float(i['value']) for i in buckets[key]], height=-100)
                meal_troughs = [x for i, x in enumerate(buckets[key]) if i in meal_trough_idxs]
                res.extend(sorted(meal_troughs, key=lambda i: float(i['value']))[:3])
            target_time = datetime.strptime(i, '%H:%M:%S')
            if int(key) == int(i.split(':')[0]):
                target_time = datetime.strptime(i, '%H:%M:%S')
                after_meal_data.extend([entry for entry in buckets[key] if datetime.strptime(entry['time'].split(' ')[1].strip(), '%H:%M:%S') >= target_time])
            if int(key) == int(i.split(':')[0]) + 2:
                target_time = datetime.strptime(i[3:].strip(), '%M:%S')
                after_meal_data.extend([entry for entry in buckets[key] if datetime.strptime(entry['time'].split(' ')[1][3:].strip(), '%M:%S') <= target_time])
            if int(key) == int(i.split(':')[0]) + 1:
                after_meal_data.extend(buckets[key])
                # if (int(key) in [int(i.split(':')[0]) + 1 for i in img_time]) or (int(key) in [int(i.split(':')[0]) + 2 for i in img_time]):
        meal_peak_idxs, _ = find_peaks([i['value'] for i in after_meal_data], height=0)
        meal_peaks = [x for i, x in enumerate(after_meal_data) if i in meal_peak_idxs]
        meal_peaks = sorted(meal_peaks, key=lambda i: float(i['value']), reverse=True)

        min_bg = ''
        max_bg = ''
        for d in after_meal_data:
            if not min_bg:
                min_bg = d['value']
            if float(d['value']) < float(min_bg):
                min_bg = d['value']
            if not max_bg:
                max_bg = d['value']
            if float(d['value']) > float(max_bg):
                max_bg = d['value']
        meals_minmax_bg[get_meal_name(i)] = [min_bg, max_bg]

        if meal_peaks and float(meal_peaks[0]['value']) > 10.0:
            res.extend(meal_peaks[:5])
        else:
            res.extend(meal_peaks[:3])

    for key in buckets.keys():
        for i in img_time:
            if (int(key) == int(i.split(':')[0]) + 1) or (int(key) == int(i.split(':')[0]) + 2):
                # if (int(key) in [int(i.split(':')[0]) + 1 for i in img_time]) or (int(key) in [int(i.split(':')[0]) + 2 for i in img_time]):
                for j in buckets[key]:
                    if i.split(':')[1] == j['time'].split(' ')[-1].split(':')[1]:
                        res.append(j)
                        break
                    elif i.split(':')[1] < j['time'].split(' ')[-1].split(':')[1]:
                        res.append(j)
                        break
        if int(key) < 6 or int(key) > 21:  # 夜间时段
            trough_idxes, _ = find_peaks([-float(i['value']) for i in buckets[key]], height=-100)
            troughs.extend([x for i, x in enumerate(buckets[key]) if i in trough_idxes])
            for i in buckets[key]:
                if not night_low:
                    night_low = i
                elif night_low['value'] > i['value']:
                    night_low = i
            if int(key) % 2 == 0:
                res.append(buckets[key][0])
        elif 5 < int(key) < 22:      # 白天时段
            peak_idxes, _ = find_peaks([i['value'] for i in buckets[key]], height=0)
            peaks.extend([x for i, x in enumerate(buckets[key]) if i in peak_idxes])
            res.append(buckets[key][0])
            # for i,x in enumerate(buckets[key]):
            #     if not day_high:
            #         day_high = x
            #     elif day_high['value'] < x['value']:
            #         day_high = x
    res.extend(sorted(peaks, key=lambda i: float(i['value']), reverse=True)[:3])
    if len(troughs) > 0:
        if float(troughs[0]['value']) < 3.9:
            res.extend(troughs[:3])
        else:
            res.append(troughs[0])
    res.append(night_low)
    # res.append(day_high)
    unique_tuples = set(tuple(sorted(d.items())) for d in res)
    res = [dict(t) for t in unique_tuples]
    res = sorted(res, key=lambda item: item.get('time', ''))
    return res, get_meals_bg_str(meals_minmax_bg)

def get_meals_bg_str(bg):
    res = '\n3. 餐后2小时内血糖极值变化：\n'
    for key in bg.keys():
        res += f'   - {key}后2小时内: 血糖最小值：{bg[key][0]}   血糖最大值：{bg[key][1]}\n'
    return res


daily_diet_eval_prompt = """# 已知信息
## 个人信息：
{0}

## 当天饮食及每餐评价信息：
{1}

## 当天1日关键血糖值：
{2}

## 当天1日动态血糖分析报告：
{3}
{4}

## 管理场景：
{5}

# 任务描述
请你扮演一位经验丰富的营养师，你正在协同医生、运动师、情志调理师、中医师，共同为慢病患者提供全方位的健康管理服务。帮助患者建立并维持健康的生活方式，例如合理饮食、适量运动、科学合理用药等，现在请你对患者提交的一日饮食信息做出合理评价和科学建议。

# 输出要求
 - 请你紧密结合患者的已知信息，如血糖数据、饮食信息、现患疾病等因素，给予患者合理的分析与建议话术。
 - 输出可能包含的2个维度有：1.血糖趋势分析、2.营养优化建议。
 - 输出的内容要通俗易懂，简单明了，符合营养学观点。
 - 每个维度的评价可以换行显示，请参考输出示例中的内容和格式，可根据不同患者情况适当调整话术内容。
 - 若血糖控制较稳定且饮食合理，可分别输出鼓励性话术。
## 血糖趋势分析输出要求
 - 血糖稳定性请参考餐前餐后血糖值波动、最低值最高值、最大血糖波动的维度来评估，说明具体的波动情况，要客观符合事实。
 - 可以指出某餐餐后血糖波动明显的情况，评价餐后血糖波动可能与选择的食物的关系，评价食物选择是否合理，可以从碳水化合物、蛋白质、脂肪的维度进行评价。
 - 请关注餐后血糖的时间段，尽量取餐后的最高值说明情况。
 - 识别异常血糖信息，并给予提醒，比如对低血糖情况应给出建议。
 - 如果没有血糖数据，该评价维度可忽略。
 - 字数不超过150字。
## 营养优化建议
 - 若血糖波动较大，血糖控制欠佳，可输出指导应该避免哪类食物，选择哪类营养素或食物的话术。
 - 若一日某餐次的饮食信息有缺失，可以给予提醒规律用餐的重要性。
 - 避免输出123列表。
 - 字数不超过100字。

# 输出参考示例
【血糖趋势分析】
监测显示你今日最大血糖值为{{num}}mmol/L，最小血糖值为{{num}}mmol/L，血糖最大波动幅度为{{num}}mmol/L。{{用餐时段}}后血糖升至{{num_high}}mmol/L，血糖波动幅度{{较大/较平稳}}，需要关注夜间低血糖风险。
分析数据表明，{{meal_time}}餐食中{{nutrient_type}}含量较高，且{{营养搭配比例欠合理}}。{{meal_time_2}}营养素构成虽有助于血糖平稳，但{{missing_nutrient}}摄入不足，可能影响血糖稳定性。
【营养优化建议】
建议调整{{meal_time}}餐食构成，选择{{low_gi_type}}类食材，适当增加{{fiber_type}}的摄入。{{meal_time_2}}可补充{{protein_type}}，注意保持主食供能。保持{{timing}}用餐规律，控制适量，有助于血糖的稳定。
可以根据具体情况填入相应的数值和营养素类型，避免具体食物示例带来的误导

Begins!"""



daily_diet_degree_prompt = """# 已知信息
## 个人信息：
{0}

## 当天饮食及每餐评价信息：
{1}

## 当天1日关键血糖值：
{2}

## 当天1日动态血糖分析报告：
{3}
{4}

## 管理场景：
{5}

# 任务描述
请你扮演一位经验丰富的营养师，请你对我提交的一日饮食信息做出当天饮食合理等级判定。

# 输出要求
- 针对用户的一日饮食情况，充分结合用户自身血糖情况，给出用户饮食合理等级。
- 饮食合理等级列表：['欠佳','尚可','极佳']，输出必须从列表中选择，禁止自己创造。

## 遵循以下格式回复:
Thought: 结合用户血糖情况和饮食情况,思考当日饮食合理等级
Output: 输出饮食合理等级

Begins!"""



diet_image_recog_prompt = """# 你扮演一名健康饮食管理助手，你需要识别出图中食物名称、数量、单位。

## 输出要求：
- 仔细分析图片，精确识别出所有可见的食材，并对每种食材进行详细的数量统计。
- 食物名称要尽可能精确到具体食材（如炒花菜、豆芽炒肉、白米饭、紫米饭等），而非泛泛的类别。
- 根据食材的特点，给出准确且恰当的数量描述和单位。例如，使用'个'来表示完整的水果（如'1个（小）苹果'、'2个橘子'），如果是一半根黄瓜则为'0.5根黄瓜'，用'片'来表示切片的食材（如'3片面包'），对于堆积的食物可以使用'堆'、'把'等（如'1堆瓜子'、'1把葡萄'），对于肉类可以用'掌心大小'、'克'、'块'等来表示分量，蔬菜类可以用'拳头大小'、'克'、'份'等来表示分量。确保所有计数均准确无误，单位使用得当。
- 输出食物必须来自图片中，禁止自己创造。
- 如果图片中有不确定种类的食材，则忽略该食材，不输出。
- 以json格式输出，严格按照`输出格式样例`形式。

## 输出格式样例：
```json
[
    {"foodname": "玉米", "count": "2", "unit": "根"},
    {"foodname": "苹果", "count": "1", "unit": "个（小）"},
    {"foodname": "苹果", "count": "1", "unit": "个（中等）"},
    {"foodname": "黄瓜", "count": "0.5", "unit": "根"},
    {"foodname": "鸡胸肉", "count": "1", "unit": "掌心大小"},
    {"foodname": "炒花菜", "count": "1", "unit": "拳头大小"},
    {"foodname": "芹菜炒肉", "count": "1", "unit": "份"},
    {"foodname": "五花肉", "count": "3", "unit": "块"},
    {"foodname": "米饭", "count": "1", "unit": "碗"},
    {"foodname": "馒头", "count": "0.5", "unit": "块"},
    {"foodname": "西红柿炒鸡蛋", "count": "1", "unit": "份"}
]
```

Begins!"""


sport_schedule_recog_prompt = """# 你是一名运动日程管理助手，你需要根据`用户日程模版信息`和`用户对话交互信息`，识别是否需要更改用户的运动日程，如果需要则给出修改原因（reason）、原因分类（category）、修改建议（suggestion）、修改天数（days）。

## 用户日程信息：
{0}

## 用户对话交互信息：
{1}

## 当前时间：
{2}

## 已知可能的需要修改运动日程的原因分类及具体内容：
category：具体内容
血糖异常：血糖数值＜4.0mmol/L、血糖数值＞14.0mmol/L
血压异常：收缩压≥160，或舒张压≥100、收缩压≤90，或舒张压≤60
存在糖尿病并发症：严重糖尿尿病肾病、视网膜病变、糖尿病足
身体不适：感冒、发烧、流感、腹泻、肠胃炎、心慌、头晕、恶心、疼痛、下肢关节扭伤、下肢关节疼痛、下肢关节酸胀、下肢肌肉拉上、骨折、上肢肌肉拉伤、上肢关节剧烈疼痛、上肢关节不适、哮喘、高血压急症、疾病康复且无静养医嘱
检查异常：心电图异常、血常规异常、尿常规异常
环境因素：空气严重污染、恶劣天气：高温、低温、高湿、紫外线强烈、大风
生活方式变化：熬夜、饮酒、疲劳、献血后
女性特殊时期：月经期、孕期、哺乳期
突发事件：出行计划改变/时间安排变化、血糖仪故障

## 输出要求：
- 首先，关注日程信息中是否有运动日程，没有运动日程则is_modify输出false，category、reason、suggestion和days都输出无。
- 其次，关注`用户对话交互信息`中是否有`已知可能的需要修改运动日程的原因分类及具体内容`中不适合运动的情况。
- 有不适合运动的情况，则is_modify输出true，并输出相应category、
reason、suggestion和days。
- 有描述不适合运动的情况，但并非用户本人的情况则is_modify输出false。
- 原因分类category需与`已知可能的需要修改运动日程的原因分类及具体内容`保持完全一致。
- reason字数在100字以内，suggestion字数在500字以内，days直接输出数字。
- 以json格式输出。

## 输出样例1：
```json
{{"is_modify": false, "category":"无,""reason": "无", "suggestion": "无", "days": "无"}}
```

## 输出样例2：
```json
{{"is_modify": true, "category":"身体不适,""reason": "识别到客户今日腿疼，3天内的运动类型任务需要进行调整。", "suggestion": "取消运动日程，观察腿部疼痛情况。不适及时就医。","days":"3"}}\n\
```

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