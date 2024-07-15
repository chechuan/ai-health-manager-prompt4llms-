from src.utils.api_protocal import *
from data.constrant import *

jiahe_userInfo_map = {
    'age':'年龄',
    'gender':'性别',
    'height':'身高',
    'weight':'体重',
    'manage_object':'管理目标',
    'disease':'现患疾病',
    'special_diet':'特殊口味偏好',
    'allergy_food':'过敏食物',
    'taste_preference':'口味偏好',
    'is_specific_menstrual_period':'是否特殊生理期',
    'constitution': '中医体质',
}


def get_userInfo_history(userInfo, history=[]):
    user_info = JiaheUserProfile().model_dump()
    for i in userInfo:
        if userInfo[i]:
            user_info[i] = userInfo[i]
    history = [
        {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
        for i in history[-5:]     # 保留最后5轮用户信息
    ]
    his_prompt = "\n".join(
        [
            ("assistant" if not i["role"] == "user" else "user")
            + f": {i['content']}"
            for i in history
        ]
    )
    info = ''
    for i in user_info.keys():
        if user_info[i] and user_info[i] != '未知':
            info += f'{jiahe_userInfo_map[i]}：{user_info[i]}\n'

    return info, his_prompt


def get_familyInfo_history(familyInfo, history):
    infos = ''
    roles = ''
    for user in familyInfo:
        user_info = JiaheUserProfile().model_dump()
        userInfo = user.get('userInfo', {})
        for i in userInfo:
            if userInfo[i]:
                user_info[i] = userInfo[i]
        info = f'{user.get("family_role", "")}的健康标签'
        roles = roles + user.get("family_role", "") + '，'
        for i in user_info.keys():
            if user_info[i] and user_info[i] != '未知':
                info += f'{jiahe_userInfo_map[i]}：{user_info[i]}\n'
        infos = infos + '\n' + info
    roles = roles[:-1] if roles[-1] == '，' else roles
    history = [
        {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
        for i in history[-5:]       # 保留最后5轮用户信息
    ]
    his_prompt = "\n".join(
        [
            ("assistant" if not i["role"] == "user" else "user")
            + f": {i['content']}"
            for i in history
        ]
    )

    return roles, infos, his_prompt