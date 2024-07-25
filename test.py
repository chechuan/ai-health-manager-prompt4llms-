def format_historical_meal_plans(historical_meal_plans: list) -> str:
    """
    将历史食谱转换为指定格式的字符串

    参数:
        historical_meal_plans (list): 包含历史食谱的列表

    返回:
        str: 格式化的历史食谱字符串
    """
    if not historical_meal_plans:
        return "历史食谱数据为空"

    formatted_output = ""

    for day in historical_meal_plans:
        date = day.get("date")
        if not date:
            continue
        formatted_output += f"{date}：\n"

        meals = day.get("meals", {})
        for meal_time, foods in meals.items():
            if not foods:
                continue
            formatted_output += f"{meal_time}：{'、'.join(foods)}\n"

    return formatted_output.strip()


# 示例数据
historical_meal_plans = [
    {
        "date": "2024-05-01",
        "meals": {
            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
            "晚餐": ["玉米", "鸡腿", "牛奶"]
        }
    },
    {
        "date": "2024-05-02",
        "meals": {
            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
            "晚餐": ["玉米", "鸡腿", "牛奶"]
        }
    },
    {
        "date": "2024-05-03",
        "meals": {
            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
            "晚餐": ["玉米", "鸡腿", "牛奶"]
        }
    },
    {
        "date": "2024-05-04",
        "meals": {
            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
            "晚餐": ["玉米", "鸡腿", "牛奶"]
        }
    },
    {
        "date": "2024-05-05",
        "meals": {
            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
            "晚餐": ["玉米", "鸡腿", "牛奶"]
        }
    },
    {
        "date": "2024-05-06",
        "meals": {
            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
            "晚餐": ["玉米", "鸡腿", "牛奶"]
        }
    }
]

# 调用方法并打印结果
formatted_output = format_historical_meal_plans(historical_meal_plans)
print(formatted_output)

