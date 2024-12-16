import pandas as pd
import numpy as np
from datetime import datetime
# import matplotlib.pyplot as plt
from typing import Dict, List, Any

# plt.rcParams['font.family'] = ['Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False


class GlucoseAnalyzer:
    def __init__(self):
        # AGP相关参数设置
        self.target_range = {
            'very_low': 3.0,  # 严重低血糖
            'low': 3.9,  # 低血糖
            'high': 10.0,  # 高血糖
            'very_high': 13.9  # 严重高血糖
        }

        # AGP评估标准
        self.standards = {
            'TIR_target': 70,  # 目标范围时间比例 >70%
            'TAR_L1_target': 20,  # 1级高血糖目标 <20%
            'TAR_L2_target': 5,  # 2级高血糖目标 <5%
            'TBR_L1_target': 4,  # 1级低血糖目标 <4%
            'TBR_L2_target': 1,  # 2级低血糖目标 <1%
            'CV_target': 36,  # 变异系数目标 ≤36%
            'GMI_target': 7,  # GMI目标 <7%
            'SDBG_target': 2.0  # 血糖标准差目标 <2.0 mmol/L
        }

    def find_glucose_periods(self, df, threshold, above=True):
        periods = []
        start_time = None

        for _, row in df.iterrows():
            if above and row['value'] > threshold:
                if start_time is None:
                    start_time = row['time']
            elif not above and row['value'] < threshold:
                if start_time is None:
                    start_time = row['time']
            else:
                if start_time is not None:
                    duration = row['time'] - start_time
                    periods.append((start_time, row['time'], duration))
                    start_time = None

        if start_time is not None:
            duration = df['time'].iloc[-1] - start_time
            periods.append((start_time, df['time'].iloc[-1], duration))

        return periods

    def calculate_sdbg(self, values: pd.Series) -> float:
        """计算血糖标准差SDBG"""
        daily_mean = values.mean()
        n = len(values)
        if n < 2:
            return 0
        squared_diff_sum = sum((x - daily_mean) ** 2 for x in values)
        return round(np.sqrt(squared_diff_sum / (n - 1)), 2)

    def calculate_agp_metrics(self, values: pd.Series) -> Dict[str, Any]:
        """计算AGP报告中的关键指标"""
        mean_glucose = values.mean()

        # 基础计算
        metrics = {
            'GMI': round(3.31 + (0.02392 * mean_glucose * 18), 2),  # 血糖管理指标
            'CV': round((values.std() / mean_glucose) * 100, 2),  # 变异系数
            'SDBG': self.calculate_sdbg(values),  # 血糖标准差
            'mean_glucose': round(mean_glucose, 2),  # 平均血糖
        }

        # 时间范围分析
        total_readings = len(values)
        metrics['time_ranges'] = {
            'TBR_L2': round(len(values[values < self.target_range['very_low']]) / total_readings * 100, 1),
            'TBR_L1': round(len(values[(values >= self.target_range['very_low']) &
                                       (values < self.target_range['low'])]) / total_readings * 100, 1),
            'TIR': round(len(values[(values >= self.target_range['low']) &
                                    (values <= self.target_range['high'])]) / total_readings * 100, 1),
            'TAR_L1': round(len(values[(values > self.target_range['high']) &
                                       (values <= self.target_range['very_high'])]) / total_readings * 100, 1),
            'TAR_L2': round(len(values[values > self.target_range['very_high']]) / total_readings * 100, 1)
        }

        return metrics

    def calculate_daily_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算每日指标"""
        df_day = df.copy()
        df_day['hour'] = df_day['time'].dt.hour

        # 基础指标计算
        metrics = {'mean_glucose': round(df['value'].mean(), 2), 'glucose_sd': round(df['value'].std(), 2),
                   'day_cv': round((df['value'].std() / df['value'].mean()) * 100, 1), 'max_glucose': df['value'].max(),
                   'max_glucose_time': df.loc[df['value'].idxmax(), 'time'], 'min_glucose': df['value'].min(),
                   'min_glucose_time': df.loc[df['value'].idxmin(), 'time'],
                   'max_delta': df['value'].max() - df['value'].min(),
                   'high_glucose_periods': self.find_glucose_periods(df, self.target_range['high']),
                   'low_glucose_periods': self.find_glucose_periods(df, self.target_range['low'], above=False)}

        # 时段分析
        time_blocks = {
            '夜间': (0, 6),
            '早餐': (6, 10),
            '午餐': (11, 14),
            '晚餐': (17, 21),
            '睡前': (21, 24)
        }

        metrics['time_blocks'] = {}
        for period, (start, end) in time_blocks.items():
            period_data = df_day[(df_day['hour'] >= start) & (df_day['hour'] < end)]['value']
            if not period_data.empty:
                metrics['time_blocks'][period] = {
                    'mean': round(period_data.mean(), 2),
                    'median': round(period_data.median(), 2),
                    'cv': round((period_data.std() / period_data.mean()) * 100, 1),
                    'sdbg': self.calculate_sdbg(period_data)
                }

        return metrics

    def assess_glucose_stability(self, metrics: Dict) -> List[str]:
        """评估血糖稳定性"""
        assessment = []

        # 评估变异系数
        if metrics['CV'] <= self.standards['CV_target']:
            assessment.append(f"血糖变异系数({metrics['CV']}%)在目标范围内,血糖控制稳定")
        else:
            assessment.append(
                f"血糖变异系数({metrics['CV']}%)超过目标值{self.standards['CV_target']}%,建议改善血糖管理")

        # 评估标准差
        if metrics['SDBG'] <= self.standards['SDBG_target']:
            assessment.append(f"血糖标准差({metrics['SDBG']})在目标范围内")
        else:
            assessment.append(f"血糖标准差({metrics['SDBG']})偏高,建议关注血糖波动")

        return assessment

    def analyze_glucose_data(self, raw_data) -> Dict[str, Any]:
        """主分析函数"""
        # 解析数据
        parsed_data = []
        for info in raw_data:
            time_str = info['time']
            value = float(info['value'])
            parsed_data.append({
                'time': datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S'),
                'value': value
            })
        df_parsed = pd.DataFrame(parsed_data)
        df_parsed.sort_values('time', inplace=True)

        # 计算所有指标
        agp_metrics = self.calculate_agp_metrics(df_parsed['value'])
        daily_metrics = self.calculate_daily_metrics(df_parsed)
        stability_assessment = self.assess_glucose_stability(agp_metrics)

        # 生成综合报告
        report = {
            '基础指标': {
                'GMI': agp_metrics['GMI'],
                '变异系数(CV)': agp_metrics['CV'],
                '标准差(SDBG)': agp_metrics['SDBG'],
                '平均血糖': agp_metrics['mean_glucose'],
                '最高血糖': daily_metrics['max_glucose'],
                '最低血糖': daily_metrics['min_glucose'],
                '血糖最大波动幅度': round(daily_metrics['max_delta'], 2)
            },
            '时间分布': agp_metrics['time_ranges'],
            '日间分析': daily_metrics,
            '稳定性评估': stability_assessment,
        }

        # 生成可视化
        # self.plot_glucose_profile(df_parsed)

        # 生成总结文本
        summary = self.generate_summary(report)

        return {
            'report': report,
            'summary': summary,
            # 'plot': plt
        }

    def generate_summary(self, report: Dict) -> str:
        """生成详细的分析总结"""
        time_ranges = report['时间分布']

        high_glucose_periods = [
            f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')} 持续{int(duration.total_seconds() / 60)}分钟"
            for start, end, duration in report['日间分析']['high_glucose_periods']]

        low_glucose_periods = [
            f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')} 持续{int(duration.total_seconds() / 60)}分钟"
            for start, end, duration in report['日间分析']['low_glucose_periods']]
        summary = f"""1. 时间分布分析:
   - 低血糖时段: {', '.join(low_glucose_periods) or '无'}
   - 高血糖时段: {', '.join(high_glucose_periods) or '无'} 

2. 血糖波动分析:  
   - 最小血糖值: {report['基础指标']['最低血糖']} mmol/L
   - 最大血糖值: {report['基础指标']['最高血糖']} mmol/L 
   - 血糖最大波动幅度: {report['基础指标']['血糖最大波动幅度']} mmol/L (目标<4.4 mmol/L)"""

        return summary


if __name__ == "__main__":
    analyzer = GlucoseAnalyzer()
    file_path = "../../../../data/bgs_1210.xlsx"
    results = analyzer.analyze_glucose_data(file_path)

    print("\n=== 一日血糖分析报告 ===")
    print(results['summary'])

    # # 显示图表
    # # 保存图表
    # plot_path = "./Data/daily_glucose_plot.png"
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # print(f"图表已保存至: {plot_path}")
    # plt.show()