import pandas as pd
import numpy as np

# 生成时间戳
start_time = '2022-01-01 00:00:00'
end_time = '2022-01-01 23:59:59'
timestamp = pd.date_range(start=start_time, end=end_time, freq='1min')

# 生成电压、电流、频率、功率的数据
voltage = np.random.normal(loc=220, scale=11/3, size=len(timestamp)) # 正态分布函数normal()，均值loc=220，标准差scale=11/3
current = np.random.normal(loc=5, scale=0.5/3, size=len(timestamp)) # 正态分布函数normal()，均值loc=5，标准差scale=0.5/3
frequency = np.random.normal(loc=50, scale=0.05, size=len(timestamp)) # 正态分布函数normal()，均值loc=50，标准差scale=0.05
power = voltage * current # 功率 = 电压 * 电流

# 将数据合并为数据框
data = pd.DataFrame({'timestamp': timestamp, 'voltage': voltage, 'current': current, 'frequency': frequency, 'power': power})
print(type(data))
# 将数据保存为CSV文件
data.to_csv('grid_data.csv', index=False)

# 将数据保存为Excel文件
# data.to_excel('grid_data.xlsx', index=False)
