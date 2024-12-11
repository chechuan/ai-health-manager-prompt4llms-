import yaml, os
from collections import OrderedDict
from scipy.signal import find_peaks
from src.pkgs.models.func_eval_model.daily_image_analysis import get_meal_increase_glucose_periods

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

def get_daily_diet_str(daily_diet_info, daily_blood_glucose):
    daily_diet_str = ''
    periods_str = get_meal_increase_glucose_periods(daily_diet_info, daily_blood_glucose)
    for idx, i in enumerate(daily_diet_info):
        diet_info = ''
        for info in i.get('diet_info', []):
            if info:
                diet_info += f"{info.get('count', '')}{info.get('unit', '')}{info.get('foodname', '')}，"
        if len(periods_str) == len(daily_diet_info):
            daily_diet_str += f"就餐时间：{i.get('diet_time', '')} 就餐食物：{diet_info}。医生当餐评价：{i.get('diet_eval', '无')}。 {periods_str[idx]}\n"
        else:
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
            img_time.append(info.get('diet_time', '').split(' ')[-1])
    night_low = {}
    day_high = {}
    peaks = []
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
            for i,x in enumerate(buckets[key]):
                if not day_high:
                    day_high = x
                elif day_high['value'] < x['value']:
                    day_high = x
    res.extend(sorted(peaks, key=lambda i: i['value'], reverse=True)[:3])
    res.append(night_low)
    # res.append(day_high)
    unique_tuples = set(tuple(sorted(d.items())) for d in res)
    res = [dict(t) for t in unique_tuples]
    res = sorted(res, key=lambda item: item.get('time', ''))
    return res


daily_diet_eval_prompt = """# 已知信息
## 个人信息：
{0}

## 当天饮食及每餐评价信息：
{1}

## 一日血糖信息：
{2}

## 管理场景：
{3}

# 任务描述
请你扮演一位经验丰富的营养师，请你对我提交的一日饮食信息做出合理评价和科学建议。

# 输出要求
 - 你需要根据我的血糖数据、提交的饮食信息以及我的已知信息如疾病等因素综合分析，来输出饮食评价和建议。
 - 输出可能包含的维度有：1.血糖稳定性评估、2.饮食待改善建议2个维度。
 - 整体字数控制在250字以内。
 - 输出的内容要通俗易懂，简单明了，符合营养学观点。
 - 每个维度的评价可以换行显示，严格参考输出示例中的内容和格式。
## 血糖稳定性评估输出要求
 - 血糖稳定性请参考餐前餐后血糖值波动、最低值最高值、最大血糖波动的维度来评估，说明具体的波动情况，要客观符合事实。
 - 可以指出餐后血糖波动可能与选择的食物的关系，评价食物选择是否合理。
 - 识别异常血糖信息，并给予提醒，比如对低血糖情况应给出建议。
 - 如果没有血糖数据，该评价维度可忽略。
## 饮食待改善建议
 - 饮食待改善建议主要针对可以优化的问题点着重指导，例如从用餐时间、用餐规律性、食物搭配等维度输出改善建议。
 - 若一日某餐次的饮食信息有缺失，可以给予提醒规律用餐的重要性。
 - 避免输出123列表。
  
# 输出示例
【血糖分析】
看了你的血糖记录，午饭后血糖升得有点高呢，到14了。虽然后来是降下来了，不过还是比较高。
你这一天血糖起伏挺大的，从早上的12到睡前的4，差了8多，得注意一下，尤其要当心晚上会不会低血糖。
我看你午饭吃红薯啦？红薯升糖比较快，再配上咸菜，这个搭配不太合适。晚饭虽然吃的清淡，主要是蔬菜，血糖是平稳，但就是营养不太够，容易导致血糖降得太低。
【饮食建议】
午饭🍚的话，建议你换成全麦面包或者燕麦，再多吃点绿叶菜。至于晚饭🐟，得加点肉或豆腐补充蛋白质，主食也不能省，来点糙米饭或全麦面包。这样营养更均衡，血糖也不会忽高忽低的。
记住🌟：每顿饭都要适量，别吃太多，这样血糖好控制。

Begins!"""

daily_diet_eval_prompt_2 = """# 已知信息
## 个人信息：
{0}

## 当天饮食及每餐评价信息：
{1}

## 当天1日动态血糖分析报告：
{2}

## 管理场景：
{3}

# 任务描述
请你扮演一位经验丰富的营养师，请你对我提交的一日饮食信息做出合理评价和科学建议。

# 输出要求
 - 你需要根据我的血糖数据、提交的饮食信息以及我的已知信息如疾病等因素综合分析，来输出饮食评价和建议。
 - 输出可能包含的维度有：1.血糖稳定性评估、2.饮食待改善建议2个维度。
 - 整体字数控制在250字以内。
 - 输出的内容要通俗易懂，简单明了，符合营养学观点。
 - 每个维度的评价可以换行显示，严格参考输出示例中的内容和格式。
## 血糖稳定性评估输出要求
 - 血糖稳定性请参考餐前餐后血糖值波动、最低值最高值、最大血糖波动的维度来评估，说明具体的波动情况，要客观符合事实。
 - 可以指出餐后血糖波动可能与选择的食物的关系，评价食物选择是否合理。
 - 识别异常血糖信息，并给予提醒，比如对低血糖情况应给出建议。
 - 如果没有血糖数据，该评价维度可忽略。
## 饮食待改善建议
 - 饮食待改善建议主要针对可以优化的问题点着重指导，例如从用餐时间、用餐规律性、食物搭配等维度输出改善建议。
 - 若一日某餐次的饮食信息有缺失，可以给予提醒规律用餐的重要性。
 - 避免输出123列表。

# 输出示例
【血糖分析】
看了你的血糖记录，午饭后血糖升得有点高呢，到14了。虽然后来是降下来了，不过还是比较高。
你这一天血糖起伏挺大的，从早上的12到睡前的4，差了8多，得注意一下，尤其要当心晚上会不会低血糖。
我看你午饭吃红薯啦？红薯升糖比较快，再配上咸菜，这个搭配不太合适。晚饭虽然吃的清淡，主要是蔬菜，血糖是平稳，但就是营养不太够，容易导致血糖降得太低。
【饮食建议】
午饭🍚的话，建议你换成全麦面包或者燕麦，再多吃点绿叶菜。至于晚饭🐟，得加点肉或豆腐补充蛋白质，主食也不能省，来点糙米饭或全麦面包。这样营养更均衡，血糖也不会忽高忽低的。
记住🌟：每顿饭都要适量，别吃太多，这样血糖好控制。

Begins!"""

daily_diet_eval_prompt_3 = """# 已知信息
## 个人信息：
{0}

## 当天饮食及每餐评价信息：
{1}

## 当前1日关键血糖值：
{2}

## 当天1日动态血糖分析报告：
{3}

## 管理场景：
{4}

# 任务描述
请你扮演一位经验丰富的营养师，请你对我提交的一日饮食信息做出合理评价和科学建议。

# 输出要求
 - 你需要根据我的血糖数据、提交的饮食信息以及我的已知信息如疾病等因素综合分析，来输出饮食评价和建议。
 - 输出可能包含的维度有：1.血糖稳定性评估、2.饮食待改善建议2个维度。
 - 整体字数控制在250字以内。
 - 输出的内容要通俗易懂，简单明了，符合营养学观点。
 - 每个维度的评价可以换行显示，严格参考输出示例中的内容和格式。
## 血糖稳定性评估输出要求
 - 血糖稳定性请参考餐前餐后血糖值波动、最低值最高值、最大血糖波动的维度来评估，说明具体的波动情况，要客观符合事实。
 - 可以指出餐后血糖波动可能与选择的食物的关系，评价食物选择是否合理。
 - 识别异常血糖信息，并给予提醒，比如对低血糖情况应给出建议。
 - 如果没有血糖数据，该评价维度可忽略。
## 饮食待改善建议
 - 饮食待改善建议主要针对可以优化的问题点着重指导，例如从用餐时间、用餐规律性、食物搭配等维度输出改善建议。
 - 若一日某餐次的饮食信息有缺失，可以给予提醒规律用餐的重要性。
 - 避免输出123列表。

# 输出示例
【血糖分析】
看了你的血糖记录，午饭后血糖升得有点高呢，到14了。虽然后来是降下来了，不过还是比较高。
你这一天血糖起伏挺大的，从早上的12到睡前的4，差了8多，得注意一下，尤其要当心晚上会不会低血糖。
我看你午饭吃红薯啦？红薯升糖比较快，再配上咸菜，这个搭配不太合适。晚饭虽然吃的清淡，主要是蔬菜，血糖是平稳，但就是营养不太够，容易导致血糖降得太低。
【饮食建议】
午饭🍚的话，建议你换成全麦面包或者燕麦，再多吃点绿叶菜。至于晚饭🐟，得加点肉或豆腐补充蛋白质，主食也不能省，来点糙米饭或全麦面包。这样营养更均衡，血糖也不会忽高忽低的。
记住🌟：每顿饭都要适量，别吃太多，这样血糖好控制。

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