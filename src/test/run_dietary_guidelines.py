import requests
import pandas as pd
import os

# 获取当前脚本的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

# 定义数据目录的路径，放在test_data目录下
file_path = os.path.join(root_dir, 'data', 'test_data', '饮食原则参数-数据跑批表 0911.xlsx')

# 加载测试数据
df = pd.read_excel(file_path, sheet_name='示例-饮食原则')

# URL
url = "http://ai-health-manager-prompt4llms.data-engine-dev.laikang.enn.cn/aigc/sanji/kangyang"


# 手动映射Excel字段到API请求中的字段
def create_user_profile(user_data):
    # 手动映射字段
    user_profile = {
        "age": user_data.get("年龄", ""),  # 年龄
        "gender": user_data.get("性别", ""),  # 性别
        "height": user_data.get("身高", ""),  # 身高
        "weight": user_data.get("体重", ""),  # 体重
        "bmi": user_data.get("BMI", ""),  # BMI
        "daily_physical_labor_intensity": user_data.get("体力劳动强度", ""),  # 体力劳动强度
        "current_diseases": user_data.get("现患疾病", ""),  # 现患疾病
        "management_goals": user_data.get("管理目标", ""),  # 管理目标
        "food_allergies": user_data.get("食物过敏", "") if user_data.get("食物过敏", "") != "无" else "",  # 食物过敏
        "taste_preferences": user_data.get("口味偏好", "") if user_data.get("口味偏好", "") != "无" else "",  # 口味偏好
        "special_physiological_period": "" if pd.isna(user_data.get("是否特殊生理期", "")) else user_data.get(
            "是否特殊生理期", ""),  # 是否特殊生理期
        "traditional_chinese_medicine_constitution": user_data.get("中医体质", "") if user_data.get("中医体质",
                                                                                                    "") != "无" else "",
        # 中医体质
        "region": user_data.get("所处地域", "")  # 所处地域
    }
    return user_profile


# 定义函数发送请求
def send_request(user_profile, medical_records):
    headers = {
        'Content-Type': 'application/json',
    }

    # 构建请求数据
    payload = {
        "intentCode": "aigc_functions_dietary_guidelines_generation",
        "user_profile": user_profile,
        "medical_records": {"present_illness_history": medical_records}
    }

    # 发送请求并获取响应
    response = requests.post(url, json=payload, headers=headers)
    print(response)

    # 处理响应
    if response.status_code == 200:
        return payload, response.json().get('items', '无结果').replace("\\n", "\n")  # 返回请求参数和结果
    else:
        return payload, f"请求失败，状态码: {response.status_code}"


# 新建DataFrame来保存结果
result_df = pd.DataFrame(columns=["用户", "参数", "结果"])

# 对每个用户生成请求数据并发送请求，保存结果
user_columns = df.columns[1:]  # 跳过 "输入参数" 列，从用户1开始遍历
for user_column in user_columns:
    user_data = df.set_index("输入参数")[user_column].to_dict()  # 以"输入参数"为索引，生成该用户的数据字典
    user_profile = create_user_profile(user_data)
    medical_records = user_data.get("病历")

    # 发送请求获取结果
    payload, result = send_request(user_profile, medical_records)

    # 创建临时 DataFrame，包含当前用户的数据
    temp_df = pd.DataFrame({
        "用户": [user_column],
        "参数": [str(payload)],
        "结果": [result]
    })

    # 使用 pd.concat 追加数据
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

    print(f"用户 {user_column} 的结果: {result}")
    print(f"用户 {user_column} 的请求参数: {payload}")

# 保存结果到 test_data 目录下
output_file_path = os.path.join(root_dir, 'data', 'test_data', '饮食原则跑批测试结果.xlsx')
result_df.to_excel(output_file_path, index=False)
print(f"测试结果已保存到 {output_file_path}")
