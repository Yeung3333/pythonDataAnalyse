import pandas as pd
import numpy as np

# 生成电压、电流的数据，分别总共生成了200个
voltage = np.random.normal(loc=220, scale=11/3, size=200) # 正态分布函数normal()，均值loc=220，标准差scale=11/3
current = np.random.normal(loc=5, scale=0.5/3, size=200) # 正态分布函数normal()，均值loc=5，标准差scale=0.5/3

# 将数据合并为数据框
data = pd.DataFrame({'voltage': voltage, 'current': current})
print(type(data))
# 将数据保存为CSV文件
data.to_csv('grid_data_TensorFlow.csv', index=False)

# 将数据保存为Excel文件
# data.to_excel('grid_data.xlsx', index=False)
