from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json


class FoodCategory(Enum):
    """食物分类"""
    STAPLE = "主食类"
    PROTEIN = "蛋白质类"
    VEGETABLE = "蔬菜类"
    FRUIT = "水果类"
    DAIRY = "乳制品类"
    SNACK = "零食点心类"
    UNKNOWN = "未分类"


class FoodGIDatabase:
    """食物升糖指数(GI)数据库"""

    def __init__(self):
        # 食物分类规则
        self.food_categories = {
            FoodCategory.STAPLE: [
                "米饭", "面", "馒头", "包子", "稀饭", "粥", "面包",
                "饼", "粉", "米", "薯", "玉米", "燕麦"
            ],
            FoodCategory.PROTEIN: [
                "肉", "鱼", "虾", "蛋", "牛", "猪", "鸡", "鸭", "豆腐",
                "豆", "虾", "蟹"
            ],
            FoodCategory.VEGETABLE: [
                "菜", "萝卜", "青", "白", "韭", "芹", "姜", "蒜",
                "瓜", "菠菜", "生菜", "西兰花"
            ],
            FoodCategory.FRUIT: [
                "果", "苹果", "香蕉", "梨", "橙", "柚", "桃", "莓",
                "葡萄", "西瓜", "芒果"
            ],
            FoodCategory.DAIRY: [
                "奶", "乳", "酸奶", "奶酪", "乳酪"
            ],
            FoodCategory.SNACK: [
                "饼干", "糕", "巧克力", "薯片", "冰淇淋", "汽水", "可乐",
                "糖", "点心"
            ]
        }

        # 食物别名映射
        self.food_aliases = {
            "米饭": "白米饭",
            "大米饭": "白米饭",
            "白饭": "白米饭",
            "馒头": "白面馒头",
            "牛奶": "全脂牛奶",
        }

        # GI数据库
        self.gi_database = {
            # 主食类
            "白米饭": 83,
            "糙米饭": 68,
            "白面包": 85,
            "全麦面包": 54,
            "白面馒头": 88,
            "通心粉": 46,
            "玉米": 55,
            "燕麦片": 42,
            "红薯": 61,

            # 蛋白质类
            "鸡蛋": 0,
            "牛肉": 0,
            "猪肉": 0,
            "鸡肉": 0,
            "鱼肉": 0,
            "豆腐": 15,

            # 乳制品类
            "全脂牛奶": 27,
            "脱脂牛奶": 32,
            "酸奶": 33,
            "奶酪": 0,

            # 水果类
            "苹果": 36,
            "香蕉": 51,
            "橙子": 43,
            "葡萄": 59,
            "西瓜": 72,

            # 蔬菜类
            "胡萝卜": 47,
            "青菜": 15,
            "菠菜": 15,
            "西兰花": 10,
            "茄子": 20
        }

    def categorize_food(self, food: str) -> FoodCategory:
        """根据食物名称判断分类"""
        for category, keywords in self.food_categories.items():
            if any(keyword in food for keyword in keywords):
                return category
        return FoodCategory.UNKNOWN

    def get_food_info(self, food: str) -> Dict:
        """获取食物的详细信息"""
        standard_name = self.food_aliases.get(food, food)
        gi_value = self.gi_database.get(standard_name, -1)
        category = self.categorize_food(food)

        return {
            'name': food,
            'standard_name': standard_name,
            'gi_value': gi_value,
            'gi_level': self._get_gi_level(gi_value),
            'category': category.value,
            'is_known': gi_value != -1
        }

    def _get_gi_level(self, gi_value: int) -> str:
        """根据GI值判断等级"""
        if gi_value == -1:
            return "未知"
        elif gi_value >= 70:
            return "高GI(≥70)"
        elif 56 <= gi_value < 70:
            return "中GI(56-69)"
        else:
            return "低GI(<56)"


def merge_glucose_periods(periods: List[str], max_gap_minutes: int = 15) -> List[str]:
    """合并临近的血糖异常时段"""
    if not periods:
        return []

    # 转换为datetime对象并排序
    time_ranges = []
    for period in periods:
        start, end = map(lambda x: datetime.strptime(x.strip(), '%H:%M'), period.split('-'))
        time_ranges.append((start, end))

    time_ranges.sort(key=lambda x: x[0])

    # 合并临近时段
    merged = []
    current_start, current_end = time_ranges[0]

    for start, end in time_ranges[1:]:
        time_gap = (start - current_end).total_seconds() / 60

        if time_gap <= max_gap_minutes:
            # 更新当前时段的结束时间
            current_end = max(current_end, end)
        else:
            # 添加当前时段并开始新时段
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    # 添加最后一个时段
    merged.append((current_start, current_end))

    # 转换回字符串格式
    return [f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}" for start, end in merged]


class MealAnalyzer:
    """餐食分析器"""

    def __init__(self):
        self.db = FoodGIDatabase()

    def analyze_meal(self, meal: Dict) -> Dict:
        """分析单餐的组成和特点"""
        components = meal['components']
        meal_time = meal['time']

        # 分析每个食物
        foods_analysis = []
        categories_found = set()
        high_gi_foods = []
        unknown_foods = []

        for food in components:
            food_info = self.db.get_food_info(food)
            foods_analysis.append(f"{food}(GI:{food_info['gi_value'] if food_info['gi_value'] != -1 else '未知'})")
            categories_found.add(food_info['category'])

            if food_info['gi_value'] >= 70:
                high_gi_foods.append(food)
            elif not food_info['is_known']:
                unknown_foods.append(food)

        return {
            'time': meal_time,
            'type': self._get_meal_type(datetime.strptime(meal_time, '%H:%M')),
            'foods_analysis': foods_analysis,
            'categories': list(categories_found),
            'high_gi_foods': high_gi_foods,
            'unknown_foods': unknown_foods,
            'is_balanced': len(categories_found) >= 3
        }

    def _get_meal_type(self, time: datetime) -> str:
        """判断餐次类型"""
        hour = time.hour
        if 6 <= hour < 10:
            return "早餐"
        elif 11 <= hour < 14:
            return "午餐"
        elif 17 <= hour < 20:
            return "晚餐"
        else:
            return "零食"


class GlucoseAnalysisReport:
    """血糖分析报告生成器"""

    @staticmethod
    def generate_report(meal_analyses: List[Dict], general_suggestions: List[str]) -> str:
        """生成分析报告"""
        report = "=== 血糖管理分析报告 ===\n\n"
        report += "【餐食分析】\n"

        for analysis in meal_analyses:
            meal_time = analysis['meal_time']
            meal_type = analysis['meal_type']
            components = analysis['components']
            glucose_period = analysis['glucose_period']
            suggestions = analysis.get('suggestions', [])

            report += f"\n■ {meal_type}（{meal_time}）\n"
            report += f"食材组成：{', '.join(components)}\n"

            if glucose_period:
                report += f"相关血糖升高时段：{glucose_period}\n"
                if suggestions:
                    report += "建议：\n"
                    for idx, suggestion in enumerate(suggestions, 1):
                        report += f"{idx}. {suggestion}\n"
                else:
                    report += "建议：建议记录更多餐食细节以提供更精确的分析\n"
            else:
                report += "血糖反应：正常范围内\n"

        report += "\n【通用建议】\n"
        for suggestion in general_suggestions:
            report += f"• {suggestion}\n"

        return report


def generate_suggestions(meal_analysis: Dict, has_glucose_issue: bool) -> List[str]:
    """根据分析生成建议"""
    suggestions = []

    if has_glucose_issue:
        if meal_analysis['high_gi_foods']:
            suggestions.extend([
                f"建议限制高GI食物({', '.join(meal_analysis['high_gi_foods'])})的摄入量",
                "可以考虑选择全谷物等低GI替代品",
                "建议调整进食顺序：先食用蔬菜和蛋白质，后食用主食"
            ])

        if meal_analysis['unknown_foods']:
            suggestions.append(
                f"建议详细记录{', '.join(meal_analysis['unknown_foods'])}的具体种类、份量和烹饪方式，"
                "这将有助于更准确地分析其对血糖的影响"
            )

        if not meal_analysis['is_balanced']:
            suggestions.append(
                "建议注意营养均衡，每餐应包含主食、蛋白质和蔬菜"
            )

        if not suggestions:
            suggestions.extend([
                "建议记录详细的用餐信息（包括食物份量和烹饪方式）",
                "注意记录餐前血糖值，这有助于判断血糖波动的原因",
                "建议观察该时段的其他因素（如压力、运动等）对血糖的影响"
            ])

    return suggestions


def analyze_diet_and_glucose(meals: List[Dict], glucose_periods: List[str]) -> str:
    """主分析函数"""
    analyzer = MealAnalyzer()
    meal_analyses = []

    # 首先合并临近的血糖异常时段
    merged_glucose_periods = merge_glucose_periods(glucose_periods)

    # 分析每一餐
    for meal in meals:
        meal_analysis = analyzer.analyze_meal(meal)

        # 检查是否存在相关的血糖异常
        has_glucose_issue = False
        relevant_periods = []  # 改为列表以存储多个相关时段
        meal_time = datetime.strptime(meal['time'], '%H:%M')

        for period in merged_glucose_periods:
            start, end = map(lambda x: datetime.strptime(x.strip(), '%H:%M'), period.split('-'))
            if meal_time <= start <= meal_time + timedelta(hours=2):
                has_glucose_issue = True
                relevant_periods.append(period)

        # 生成建议
        suggestions = generate_suggestions(meal_analysis, has_glucose_issue)

        meal_analyses.append({
            'meal_time': meal['time'],
            'meal_type': meal_analysis['type'],
            'components': meal_analysis['foods_analysis'],
            'glucose_period': '; '.join(relevant_periods) if relevant_periods else None,
            'suggestions': suggestions,
            'categories': meal_analysis['categories']
        })

    # 生成通用建议
    general_suggestions = [
        "建议每日进行餐前和餐后2小时血糖监测",
        "保持规律运动，每周至少150分钟中等强度有氧运动",
        "注意饮食均衡，每餐主食量建议控制在25-30克碳水化合物",
        "建议使用血糖监测仪实时了解血糖变化",
        "对于未知GI值的食物，建议从小份量尝试开始，观察血糖反应"
    ]

    # 生成报告
    return GlucoseAnalysisReport.generate_report(meal_analyses, general_suggestions)


# 测试代码
if __name__ == "__main__":
    # 测试数据
    meals = [
        {'content': '白米饭,炒菠菜', 'components': ['白米饭', '菠菜'], 'time': '13:00'},
        {'content': '全麦面包,牛奶', 'components': ['全麦面包', '牛奶'], 'time': '07:45'},
        {'content': '红烧肉,青菜,米饭', 'components': ['红烧肉', '青菜', '米饭'], 'time': '18:50'},
        {'content': '酸奶', 'components': ['酸奶'], 'time': '15:20'}
    ]

    glucose_periods = [
        '13:24 - 13:25',
        '13:50 - 14:20',
        '15:10 - 15:17',
        '19:56 - 20:01'
    ]

    # 生成分析报告
    report = analyze_diet_and_glucose(meals, glucose_periods)
    print(report)