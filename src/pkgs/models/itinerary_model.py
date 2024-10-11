from datetime import datetime, timedelta

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
