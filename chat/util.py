import re

nation_list = ["彝族","壮族","布依族","朝鲜族","满族","侗族","瑶族","白族","土家族","哈尼族","哈萨克族","傣族","黎族","傈僳族","佤族","畲族","高山族","拉祜族","水族","东乡族","纳西族","景颇族","柯尔克孜族","土族","达斡尔族","仫佬族","羌族","布朗族","撒拉族","毛难族","仡佬族","锡伯族","阿昌族","普米族","塔吉克族","怒族","乌孜别克族","俄罗斯族","鄂温克族","德昂族","保安族","裕固族","京族","塔塔尔族","独龙族","鄂伦春族","赫哲族","门巴族","珞巴族","基诺族","未知","呼叫转移","停止"]
dis_list = ["高血压","糖尿病","高尿酸血症/痛风","高同型半胱氨酸血症","高脂血症","呼吸系统疾病","心脏病","肝病","肾病","皮肤病","血液病","恶性肿瘤","其他","无现患疾病","未知","呼叫转移","停止"]
family_dis_list = ["糖尿病家族史","冠心病家族史","高血压家族史","痛风/高尿酸血症家族史","脑卒中家族史","青光眼家族史","偏头痛家族史","精神心理类疾病家族史","无家族病史","慢性阻塞性肺疾病家族史","肝炎家族史","恶性肿瘤家族史","结核病病史","肾炎家族史","脂肪肝家族史","贫血家族史","甲状腺疾病家族史","心脏病家族史","胃癌家族史","胃溃疡家族史","其他","未知","呼叫转移","停止"]
goal_list = ["减脂","塑形","保持体重","增重","血压管理","尿酸管理","血脂管理","血糖管理","控制同型半胱氨酸","提高心肺耐力","矫正不良体姿","提高关节活动度","非以上管理目标","未知","呼叫转移","停止"]
taste_prefer_list = ["口味清淡","喜甜","喜咸","喜辣","喜酸","口味接近江浙沪","口味接近西北","口味接近东北","口味接近川湘","口味接近云贵","口味接近粤","口味接近京津","口味接近山东","口味无特殊偏好","未知","呼叫转移","停止"]
food_alergy_list = ["奶类过敏","蛋类过敏","大豆过敏","麸质过敏","花生过敏","坚果过敏","鱼类过敏","虾类过敏","贝类过敏","海鲜过敏","芒果过敏","无过敏食物","未知","呼叫转移","停止"]
sport_habbit_list = ["无运动习惯","每周运动1-2次","每周运动3-4次","每周运动≥5次","未知","呼叫转移","停止"]
degree_list = ["颈部轻、中度疼痛","颈部重度疼痛","腰部轻、中度疼痛","腰部重度疼痛","臀部轻、中度疼痛","臀部重度疼痛","膝关节轻、中度疼痛","膝关节重度疼痛","踝关节轻、中度疼痛","踝关节重度疼痛","肩部轻、中度疼痛","肩部重度疼痛","腕部轻、中度疼痛","腕部重度疼痛","无运动禁忌","未知","呼叫转移","停止"]
chest_pain_list = ["有过胸痛","无胸痛","未知","呼叫转移","停止"]
mmol_drug_list = ["糖尿病未用药，病情稳定","糖尿病在用胰岛素，病情稳定","糖尿病在用胰岛素，病情多变","糖尿病未用药，病情多变","糖尿病在用药，非胰岛素，病情稳定","糖尿病在用药，非胰岛素，病情多变","未知","呼叫转移","停止"]
labour_list = ["体力劳动强度极轻","体力劳动强度轻","体力劳动强度中","体力劳动强度高","体力劳动强度极高","未知","呼叫转移","停止"]

def get_mmol_drug(content):
    for i in mmol_drug_list:
        if i in content:
            return i
    return '未知'

def get_chest_pain(content):
    for i in chest_pain_list:
        if i in content:
            return i
        else:
            return '未知'

def get_degree_pain(content):
    for i in degree_list:
        if i in content:
            return i
    return '未知'

def get_sport_habbit(content):
    for i in sport_habbit_list:
        if i in content:
            return i
    return '未知'

def get_food_alergy(content):
    ret = ''
    for i in food_alergy_list:
        if i in content:
            ret += i
    if ret:
        return ret
    else:
        return "未知"

def get_tast_prefer(content):
    ret = ''
    for taste in taste_prefer_list:
        if taste in content:
            ret += taste
    if ret:
        return ret
    else:
        return "未知"

def get_sex(content):
    if '男' in content:
        return '男'
    elif '女' in content:
        return '女'
    else:
        return '未知'
    
def get_age(content):
    if re.search(r"(\d{4}-\d{1,2}-\d{1,2})",content):
        return re.search(r"(\d{4}-\d{1,2}-\d{1,2})",content).group(0)
    elif re.search(r"(\d{4}-\d{1,2})",content):
        return re.search(r"(\d{4}-\d{1,2}-\d{1,2})",content).group(0)
    elif re.search(r"(\d{4})",content):
        return re.search(r"(\d{4}-\d{1,2}-\d{1,2})",content).group(0)
    else:
        return '未知'
    
def get_nation(content):
    for nation in nation_list:
        if nation in content:
            return nation
    return '未知'

def get_height(content):
    height = re.search(r"\d+",content).group(0)
    if height:
        return height
    else:
        return '未知'
    
def get_weight(content):
    weight = re.search(r"\d+",content).group(0)
    if weight: 
        return weight
    else:   
        return '未知'

def get_labour_intentisy(content):
    for i in labour_list:
        if i in content:
            return i
    return '未知'
    
def get_disease(content):
    ret = ''
    for i in dis_list:
        if i in content:
            ret += i
    if ret:
        return ret
    else:
        return "未知"
    
def get_family_disease(content):
    ret = ''
    for i in family_dis_list:
        if i in content:
            ret += i
    if ret:
        return ret
    else:
        return "未知"

def get_goal_manage(content):
    for goal in goal_list:
        if goal in content:
            return goal
    return '未知'

def get_name(content):
    return content if content else '未知'


def norm_userInfo_msg(intentCode, content):
    if intentCode == 'ask_six':
        return get_sex(content)
    elif intentCode == 'ask_name':
        return get_name(content)
    elif intentCode == 'ask_age':
        return get_age(content)
    elif intentCode == 'ask_nation':
        return get_nation(content)
    elif intentCode == 'ask_height':
        return get_height(content)
    elif intentCode == 'ask_weight':
        return get_weight(content)
    elif intentCode == 'ask_labor_intensity':
        return get_labour_intentisy(content)
    elif intentCode == 'ask_disease':
        return get_disease(content)
    elif intentCode == 'ask_family_history':
        return get_family_disease(content)
    elif intentCode == 'ask_goal_manage':
        return get_goal_manage(content)
    elif intentCode == 'ask_mmol_drug':
        return get_mmol_drug(content)
    elif intentCode == 'ask_food_alergy':
        return get_food_alergy(content)
    elif intentCode == 'ask_taste_prefer':
        return get_tast_prefer(content)
    elif intentCode == 'ask_exercise_habbit':
        return get_exercise_habbit(content)
    elif intentCode == 'ask_exercise_taboo_degree':
        return get_degree_pain(content)
    elif intentCode == 'ask_exercise_habbit':
        return get_sport_habbit(content)
    elif intentCode == 'ask_exercise_taboo_xt':
        return get_chest_pain(content)
    else:
        return '未知'
