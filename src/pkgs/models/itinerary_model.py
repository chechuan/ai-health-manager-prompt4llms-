from fastapi import FastAPI, Request
from typing import List, Optional
import random
from datetime import datetime, timedelta
import pandas as pd
import re

app = FastAPI()

class ItineraryGenerator:
    def __init__(self):
      # 初始化可以用于未来扩展的部分
      pass

    def generate(self, user_data):
      """
      根据用户数据生成简单的行程清单
      :param user_data: 用户的输入数据，包括偏好、需求等
      :return: 行程清单的响应字典
      """
      # 将 start_date 和 end_date 转换为 datetime 对象
      # start_date = datetime.strptime(user_data['service_time']['start_date'], "%Y-%m-%d")
      # end_date = datetime.strptime(user_data['service_time']['end_date'], "%Y-%m-%d")
      #
      # total_days = (end_date - start_date).days + 1
      # itinerary = []
      #
      # # 根据用户数据生成行程
      # for day in range(1, total_days + 1):
      #     itinerary.append({
      #         "day": day,
      #         "date": (start_date + timedelta(days=day-1)).strftime("%Y-%m-%d"),
      #         "time_slots": self.get_time_slots(day)
      #     })

      return {
        "head": 200,
        "items": {
          "hotel": {
            "name": "汤泉逸墅",
            "extra_info": {
              "location": "具体地址信息待定",
              "description": "房型丰富、设施齐全、中医理疗特色",
              "link": "待定"
            }
          },
          "recommendation_basis": "推荐内容基于用户输入的健康状况、偏好及其他条件，匹配合适的酒店和行程方案。",
          "itinerary": [
            {
              "day": 1,
              "date": "2024-10-27",
              "time_slots": [
                {
                  "period": "下午",
                  "activities": [
                    {
                      "name": "办理入住",
                      "location": "汤泉逸墅",
                      "extra_info": {
                        "description": "房型丰富、设施齐全、中医理疗特色",
                        "operation_tips": "提醒需要预约等",
                        "activity_link": "待定"
                      }
                    },
                    {
                      "name": "温泉疗愈",
                      "location": "汤泉逸墅",
                      "extra_info": {
                        "description": "温泉疗愈服务，适合放松身心。",
                        "activity_link": "待定"
                      }
                    }
                  ]
                }
              ]
            },
            {
              "day": 2,
              "date": "2024-10-28",
              "time_slots": [
                {
                  "period": "中午",
                  "activities": [
                    {
                      "name": "岩洞氧吧体验",
                      "location": "岩洞氧吧",
                      "extra_info": {
                        "description": "体验天然氧吧，舒缓压力。",
                        "activity_link": "待定"
                      }
                    }
                  ]
                },
                {
                  "period": "下午",
                  "activities": [
                    {
                      "name": "果蔬采摘",
                      "location": "来康郡庄园",
                      "extra_info": {
                        "description": "草莓、蓝莓、葡萄等果蔬采摘体验",
                        "activity_link": "待定"
                      }
                    },
                    {
                      "name": "果树认领",
                      "location": "来康郡庄园",
                      "extra_info": {
                        "description": "选择个人专属认证树果",
                        "activity_link": "待定"
                      }
                    },
                    {
                      "name": "劳作体验",
                      "location": "来康郡庄园",
                      "extra_info": {
                        "description": "花卉的种植养护",
                        "activity_link": "待定"
                      }
                    }
                  ]
                }
              ]
            },
            {
              "day": 3,
              "date": "2024-10-29",
              "time_slots": [
                {
                  "period": "上午",
                  "activities": [
                    {
                      "name": "香修体验",
                      "location": "七修书院",
                      "extra_info": {
                        "description": "玲珑香囊制作",
                        "activity_link": "待定"
                      }
                    },
                    {
                      "name": "花修体验",
                      "location": "七修书院",
                      "extra_info": {
                        "description": "中式插花",
                        "activity_link": "待定"
                      }
                    }
                  ]
                },
                {
                  "period": "下午",
                  "activities": [
                    {
                      "name": "食修体验",
                      "location": "七修书院",
                      "extra_info": {
                        "description": "微酿自酿",
                        "activity_link": "待定"
                      }
                    },
                    {
                      "name": "功修体验",
                      "location": "七修书院",
                      "extra_info": {
                        "description": "导引养生功系列",
                        "activity_link": "待定"
                      }
                    },
                    {
                      "name": "七修坊",
                      "location": "七修书院",
                      "extra_info": {
                        "description": "午餐",
                        "activity_link": "待定"
                      }
                    },
                    {
                      "name": "萌宠互动体验",
                      "location": "萌宠乐园",
                      "extra_info": {
                        "description": "与萌宠互动的乐趣体验",
                        "activity_link": "待定"
                      }
                    }
                  ]
                }
              ]
            }
          ],
          "msg": "行程生成成功"
        }
      }

    def get_time_slots(self, day):
      """
      获取示例活动（根据天数写死的简单活动）
      :param day: 当前是第几天
      :return: 活动列表
      """
      if day == 1:
        return [
          {
            "period": "下午",
            "activities": [
              {"name": "办理入住", "location": "汤泉逸墅", "extra_info": {"note": "需提前预约"}},
              {"name": "温泉疗愈", "location": "汤泉逸墅", "extra_info": {"description": "放松身心"}}
            ]
          }
        ]
      elif day == 2:
        return [
          {
            "period": "中午",
            "activities": [
              {"name": "岩洞氧吧体验", "location": "岩洞氧吧", "extra_info": {"description": "舒缓压力"}}
            ]
          },
          {
            "period": "下午",
            "activities": [
              {"name": "果蔬采摘", "location": "来康郡庄园", "extra_info": {"description": "体验乡村劳作"}},
              {"name": "劳作体验", "location": "来康郡庄园", "extra_info": {"description": "农业体验"}}
            ]
          }
        ]
      else:
        return [{"period": "全天", "activities": [{"name": "自由活动", "location": "酒店", "extra_info": {}}]}]

    # 读取温泉方案 Excel 文件
    import pandas as pd

    def load_bath_plan_data(self):
        # 文件路径
        file_path = 'doc/bath_plan/温泉方案_updated.xlsx'  # 更新为你的文件路径

        # 读取泡浴方案和温泉作用
        df_bath_plan = pd.read_excel(file_path, sheet_name='温泉方案')
        df_spring_effects = pd.read_excel(file_path, sheet_name='温泉作用')

        # 创建一个字典来存储问题组合和方案
        bath_plan_data = {}
        # 创建一个字典来存储温泉作用
        spring_effects_data = {}

        # 遍历温泉作用数据，存储每个温泉的作用
        for index, row in df_spring_effects.iterrows():
            spring_name = row['分类']  # 温泉名称
            effect = row['作用']  # 温泉的作用
            if pd.notna(spring_name) and pd.notna(effect):
                spring_effects_data[spring_name] = effect

        # 遍历泡浴方案数据，存储每个问题组合对应的泡浴方案
        for index, row in df_bath_plan.iterrows():
            question_combination = row['问题组合']  # 确保 Excel 中这一列是问题组合
            bath_plan = row['泡浴方案']  # 确保 Excel 中这一列是泡浴方案

            # 只处理非空组合和方案
            if pd.notna(question_combination) and pd.notna(bath_plan):
                bath_plan_data[question_combination] = bath_plan

        return bath_plan_data, spring_effects_data

    def parse_bath_plan(self, bath_plan_string, spring_effects_data):
        # 固定主池和副池的时间
        main_pool_time = "建议15-20分钟"
        secondary_pool_time = "建议10-15分钟"

        # 使用正则表达式提取温泉名和主池标记
        bath_plan = []
        pattern = r'(\w+泉)(?:（(主池)）)?'
        matches = re.findall(pattern, bath_plan_string)

        # 遍历匹配到的温泉名和池类型
        for match in matches:
            spring_name = match[0]
            pool_type = "主池" if match[1] == '主池' else "副池"

            # 设置固定的时间
            suggested_time = main_pool_time if pool_type == "主池" else secondary_pool_time

            # 获取温泉的作用
            effect = spring_effects_data.get(spring_name, "未知作用")

            # 构建温泉对象，包含作用
            bath_plan.append({
                "spring_name": spring_name,
                "pool_type": pool_type,
                "suggested_time": suggested_time,
                "effect": effect
            })

        return bath_plan

    def generate_bath_plan(self, user_data):
        bath_plan_data, spring_effects_data = self.load_bath_plan_data()

        full_plan = []
        user_problems = []

        # 根据用户健康问题生成组合
        if user_data.get('skin_problems'):
            user_problems.append('皮肤问题')

        if user_data.get('pain_problems'):
            user_problems.append('疼痛问题')

        if user_data.get('fatigue_problems'):
            user_problems.append('疲劳问题')

        if user_data.get('sleep_problems'):
            user_problems.append('睡眠问题')

        # 生成问题组合的键
        if len(user_problems) == 1:
            combination_key = user_problems[0]
        elif len(user_problems) > 1:
            combination_key = '+'.join(user_problems)
        else:
            return {"msg": "无健康问题数据"}

        # 根据组合键查找方案
        if combination_key in bath_plan_data:
            bath_plan_string = bath_plan_data[combination_key]
            # 解析泡浴方案字符串并关联温泉作用
            full_plan = self.parse_bath_plan(bath_plan_string, spring_effects_data)
        else:
            return {"msg": "未找到匹配的泡浴方案"}

        return {
            "head": 200,
            "items": {
                "bath_plan": full_plan,
                "notice": "建议累计泡浴时长40-50分钟/次，避免长时间泡浴导致疲劳、低血糖等不适",
                "output_basis": "根据您的问卷结果，结合特色温泉的不同功效，从整体有效性、合理性、提升效果的角度为您推荐以下泡浴方案，仅供参考",
                "health_analysis": "您可能存在阴阳失调、肝气郁结的问题，可能与情志不畅、长期久坐，压力大有关，建议注重情绪调节，保持心情舒畅，避免长期久坐，适度进行散步等放松运动"
            },
            "msg": ""
        }

# 测试输入数据
input_data = {
    "service_theme": "旅游度假",
    "service_time": {
        "start_date": "2024-10-27",
        "end_date": "2024-10-28"
    },
    "travelers": [
        {"age_group": "adult", "count": 2},
        {"age_group": "elderly", "count": 1}
    ],
    "service_preference": ["休闲", "文化"],
    "remarks": "行程不要太紧凑，喜欢温泉和文化体验",
    "preferred_package": "豪华家庭套餐",
    "budget_range": 1000
}

# 实例化 ItineraryGenerator 类并生成行程
generator = ItineraryGenerator()
recommended_itinerary = generator.generate(input_data)

# 打印推荐结果
# print(recommended_itinerary)
