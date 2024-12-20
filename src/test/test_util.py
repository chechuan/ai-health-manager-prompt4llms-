from datetime import datetime

def find_rising_periods(data, fmt="%H:%M"):
    glucose_periods = []
    i = 0
    while i < len(data) - 1:
        start_time = datetime.strptime(data[i]['time'], fmt)
        end_time = datetime.strptime(data[i + 1]['time'], fmt)
        if data[i + 1]['glucose'] > data[i]['glucose']:
            start = i
            while i < len(data) - 1 and data[i + 1]['glucose'] > data[i]['glucose']:
                i += 1
            end = i
            glucose_periods.append(f"{data[start]['time']} - {data[end]['time']}")
        i += 1
    return glucose_periods

data = [
    {'time': '13:24', 'glucose': 100},
    {'time': '13:25', 'glucose': 102},
    {'time': '13:26', 'glucose': 98},
    {'time': '13:27', 'glucose': 100},
    {'time': '13:28', 'glucose': 105},
    {'time': '13:29', 'glucose': 100},
    {'time': '13:50', 'glucose': 90},
    {'time': '14:00', 'glucose': 95},
    {'time': '14:10', 'glucose': 100},
    {'time': '14:20', 'glucose': 105},
    {'time': '15:10', 'glucose': 100},
    {'time': '15:11', 'glucose': 102},
    {'time': '15:12', 'glucose': 105},
    {'time': '15:13', 'glucose': 103},
    {'time': '15:14', 'glucose': 105},
    {'time': '15:15', 'glucose': 108},
    {'time': '15:16', 'glucose': 110},
    {'time': '15:17', 'glucose': 108},
    {'time': '16:00', 'glucose': 110},
    {'time': '16:01', 'glucose': 112},
    {'time': '16:02', 'glucose': 110},
    {'time': '19:56', 'glucose': 100},
    {'time': '19:57', 'glucose': 102},
    {'time': '19:58', 'glucose': 105},
    {'time': '19:59', 'glucose': 103},
    {'time': '20:00', 'glucose': 105},
    {'time': '20:01', 'glucose': 108},
]

print(find_rising_periods(data))