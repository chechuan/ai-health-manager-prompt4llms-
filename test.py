import gzip
import json


def parse_weather_data(data):
    # 解析JSON数据
    data = json.loads(data)

    # 提取所需信息
    temp = data['now']['temp']
    wind_scale = data['now']['windScale']
    humidity = data['now']['humidity']
    precip = data['now']['precip']
    pressure = data['now']['pressure']
    vis = data['now']['vis']
    cloud = data['now']['cloud']
    dew = data['now']['dew']
    wind_dir = data['now']['windDir']
    weather_text = data['now']['text']

    # 构建天气描述句子
    weather_description = (
        f"今日北京天气{weather_text}，"
        f"温度{temp}摄氏度，"
        f"风力等级{wind_scale}，"
        f"相对湿度{humidity}%，"
        f"降水量{precip}毫米，"
        f"大气压强{pressure}百帕，"
        f"能见度{vis}公里，"
        f"云量{cloud}%，"
        f"露点温度{dew}摄氏度，"
        f"风向{wind_dir}。"
    )

    return weather_description


# 示例JSON数据
data = """
{
  "code": "200",
  "updateTime": "2020-06-30T22:00+08:00",
  "fxLink": "http://hfx.link/2ax1",
  "now": {
    "obsTime": "2020-06-30T21:40+08:00",
    "temp": "24",
    "feelsLike": "26",
    "icon": "101",
    "text": "多云",
    "wind360": "123",
    "windDir": "东南风",
    "windScale": "1",
    "windSpeed": "3",
    "humidity": "72",
    "precip": "0.0",
    "pressure": "1003",
    "vis": "16",
    "cloud": "10",
    "dew": "21"
  },
  "refer": {
    "sources": [
      "QWeather",
      "NMC",
      "ECMWF"
    ],
    "license": [
      "QWeather Developers License"
    ]
  }
}
"""

weather_description = parse_weather_data(data)
print(weather_description)
