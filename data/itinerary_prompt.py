# 示例提示词
# 用户需求：
#
# 行程主题：旅游度假
# 服务时间：从2024年10月27日到2024年10月29日
# 出行人员：2名成年人，1名儿童
# 用户的服务偏好：修身养性、健康调理
# 预算：控制在5000元以内
# 活动列表（根据初步筛选）：
#
# 花修体验：
#
# 费用：138元
# 适用人群：成人女性
# 描述：插花体验，提升艺术感知和审美。
# 时长：2小时
# 最佳时段：全天
# 香修体验：
#
# 费用：58元
# 适用人群：6周岁以上全员
# 描述：了解香学文化，制作合香香品。
# 时长：1.5小时
# 最佳时段：全天
# 温泉疗愈：
#
# 费用：200元
# 适用人群：全员
# 描述：温泉疗愈服务，适合放松身心。
# 时长：2小时
# 最佳时段：下午
# 任务要求：
#
# 请根据上述用户的偏好和需求，结合筛选出的活动列表，生成一个详细的3天行程推荐。行程安排应满足以下要求：
#
# 时间安排：根据每天的上午、下午和晚上时间段，合理安排不同活动，确保时间不冲突，且行程舒缓不紧凑。
# 活动安排：为不同年龄段的出行人员选择合适的活动，避免重复安排，且确保每天有适合各类人群的活动。
# 健康与娱乐结合：行程中应包含修身养性、健康调理等相关活动，如温泉疗愈、香修体验等。
# 预算控制：确保行程推荐活动的总预算控制在5000元以内。
# 请输出符合上述需求的JSON格式的行程方案。
#
# 期望输出格式：
# 每天按上午、下午、晚上时间段安排活动；
# 包含活动的名称、费用、时长、适用人群和活动描述；
# 输出内容需符合预算和时间安排。
#
# 输出示例：
#
# {
#     "head": 200,
#     "items": {
#         "hotel": {
#             "name": "汤泉逸墅",
#             "extra_info": {
#                 "location": "具体地址信息待定",
#                 "description": "房型丰富、设施齐全、中医理疗特色",
#                 "link": "待定"
#             }
#         },
#         "recommendation_basis": "推荐内容基于用户输入的健康状况、偏好及其他条件，匹配合适的酒店和行程方案。",
#         "itinerary": [
#             {
#                 "day": 1,
#                 "date": "2024-10-27",
#                 "time_slots": [
#                     {
#                         "period": "下午",
#                         "activities": [
#                             {
#                                 "name": "办理入住",
#                                 "location": "汤泉逸墅",
#                                 "extra_info": {
#                                     "description": "房型丰富、设施齐全、中医理疗特色",
#                                     "operation_tips": "提醒需要预约等"
#                                 }
#                             },
#                             {
#                                 "name": "温泉疗愈",
#                                 "location": "汤泉逸墅",
#                                 "extra_info": {
#                                     "description": "温泉疗愈服务，适合放松身心。"
#                                 }
#                             }
#                         ]
#                     }
#                 ]
#             },
#             {
#                 "day": 2,
#                 "date": "2024-10-28",
#                 "time_slots": [
#                     {
#                         "period": "中午",
#                         "activities": [
#                             {
#                                 "name": "岩洞氧吧体验",
#                                 "location": "岩洞氧吧",
#                                 "extra_info": {
#                                     "description": "体验天然氧吧，舒缓压力。"
#                                 }
#                             }
#                         ]
#                     },
#                     {
#                         "period": "下午",
#                         "activities": [
#                             {
#                                 "name": "果蔬采摘",
#                                 "location": "来康郡庄园",
#                                 "extra_info": {
#                                     "description": "草莓、蓝莓、葡萄等果蔬采摘体验"
#                                 }
#                             },
#                             {
#                                 "name": "果树认领",
#                                 "location": "来康郡庄园",
#                                 "extra_info": {
#                                     "description": "选择个人专属认证树果"
#                                 }
#                             },
#                             {
#                                 "name": "劳作体验",
#                                 "location": "来康郡庄园",
#                                 "extra_info": {
#                                     "description": "花卉的种植养护"
#                                 }
#                             }
#                         ]
#                     }
#                 ]
#             },
#             {
#                 "day": 3,
#                 "date": "2024-10-29",
#                 "time_slots": [
#                     {
#                         "period": "上午",
#                         "activities": [
#                             {
#                                 "name": "香修体验",
#                                 "location": "七修书院",
#                                 "extra_info": {
#                                     "description": "玲珑香囊制作"
#                                 }
#                             },
#                             {
#                                 "name": "花修体验",
#                                 "location": "七修书院",
#                                 "extra_info": {
#                                     "description": "中式插花"
#                                 }
#                             }
#                         ]
#                     },
#                     {
#                         "period": "下午",
#                         "activities": [
#                             {
#                                 "name": "食修体验",
#                                 "location": "七修书院",
#                                 "extra_info": {
#                                     "description": "微酿自酿"
#                                 }
#                             },
#                             {
#                                 "name": "功修体验",
#                                 "location": "七修书院",
#                                 "extra_info": {
#                                     "description": "导引养生功系列"
#                                 }
#                             },
#                             {
#                                 "name": "七修坊",
#                                 "location": "七修书院",
#                                 "extra_info": {
#                                     "description": "午餐"
#                                 }
#                             },
#                             {
#                                 "name": "萌宠互动体验",
#                                 "location": "萌宠乐园",
#                                 "extra_info": {
#                                     "description": "与萌宠互动的乐趣体验"
#                                 }
#                             }
#                         ]
#                     }
#                 ]
#             }
#         ],
#         "msg": ""
#     }
# }


import openai # Version: 1.38.0

DEFAULT_MODEL = "Qwen2-7B-Instruct"

client = openai.OpenAI(api_key="sk-QZqVzsbLfql5texCBb27C4958e2b412798F2Ba0e34D958C5", base_url="http://10.39.91.251:40024/v1/")
prompt = "请详细描述机器学习中的过拟合现象及其解决方法。"
kwds = {
    "model": DEFAULT_MODEL,
    "prompt": prompt,
    "top_p": 1.0,
    "max_tokens": 1024,
    "temperature": 0.7,
}
completion = client.completions.create(**kwds)
print("响应内容:", completion.choices[0].text)