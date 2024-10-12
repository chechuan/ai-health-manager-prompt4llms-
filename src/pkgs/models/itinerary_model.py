from fastapi import FastAPI, Request
from typing import List, Optional
import random
from datetime import datetime, timedelta

app = FastAPI()

# 数据部分 - 每种问题对应的池子方案及时长（10-15分钟）
bath_plan_data = {
    "skin_problems": [
        {"name": "硫磺泉", "time": random.randint(10, 15), "type": "主池"},
        {"name": "铁泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "通络泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "气血泉", "time": random.randint(10, 15), "type": "副池"},
    ],
    "pain_problems": [
        {"name": "碳酸泉", "time": random.randint(10, 15), "type": "主池"},
        {"name": "富氢泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "铁泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "疼痛调理泉", "time": random.randint(10, 15), "type": "副池"},
    ],
    "fatigue_problems": [
        {"name": "碳酸泉", "time": random.randint(10, 15), "type": "主池"},
        {"name": "健脾泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "气血泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "温补泉", "time": random.randint(10, 15), "type": "副池"},
    ],
    "sleep_problems": [
        {"name": "富氢泉", "time": random.randint(10, 15), "type": "主池"},
        {"name": "睡眠调理泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "通络泉", "time": random.randint(10, 15), "type": "副池"},
        {"name": "温补泉", "time": random.randint(10, 15), "type": "副池"},
    ]
}


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

    def generate_bath_plan(self, user_data):
        """
        生成泡浴方案
        :param intentcode: 用于区分用户请求类型的意图码 aigc_functions_generate_bath_plan
        :param user_data: 用户的健康问题数据
        :return: 泡浴方案
        """
        full_plan = []
        if user_data.get('skin_problems'):
            skin_plan = random.sample(bath_plan_data["skin_problems"], len(bath_plan_data["skin_problems"]))
            full_plan.extend(skin_plan)

        if user_data.get('pain_problems'):
            pain_plan = random.sample(bath_plan_data["pain_problems"], len(bath_plan_data["pain_problems"]))
            full_plan.extend(pain_plan)

        if user_data.get('fatigue_problems'):
            fatigue_plan = random.sample(bath_plan_data["fatigue_problems"], len(bath_plan_data["fatigue_problems"]))
            full_plan.extend(fatigue_plan)

        if user_data.get('sleep_problems'):
            sleep_plan = random.sample(bath_plan_data["sleep_problems"], len(bath_plan_data["sleep_problems"]))
            full_plan.extend(sleep_plan)

        random.shuffle(full_plan)

        return {
            "head": 200,
            "item": {
                "bath_plan": full_plan
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
