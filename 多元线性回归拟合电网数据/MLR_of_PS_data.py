from numpy import genfromtxt
import numpy as np
from sklearn import linear_model

datapath = r"grid_data.csv" # 读取数据文件
data = genfromtxt(datapath,delimiter=",") # genfromtxt() 是 NumPy 中的一个函数，它可以用来读取 CSV 文件（逗号分隔值文件）或其他文本文件，并将其转换为 NumPy 数组。下面是 genfromtxt() 的基本用法：

x = data[1:,:-1] # 你看这里，起步是1，之前那个是[:,:-1]，因此可以推断，这里起步的1就是意味着从第1行也就是真实的第二行开始读取
y = data[1:,-1]
print ("1.输出数据集x:\n",x,"\n") # 输出数据集x
print ("2.输出数据集y:\n",y,"\n") # 输出数据集y

mlr = linear_model.LinearRegression() # 定义线性回归函数，我不在乎你是不是多元，反正我先定义

mlr.fit(x, y) # 拟合变量x，y
print("3.系数:",mlr.coef_,"\n") # 打印系数
print("4.截距:",mlr.intercept_,"\n") # 打印截距，我估计默认是纵轴截距


# 输入预测数据集
voltage = np.random.normal(220, 11) # 电压的波动在±5%，220*0.05=11
current = np.random.normal(5,0.5) # 电流的波动在±10%，5*0.1=0.5
frequence = np.random.normal(50,0.05) # 国标波动等级规定如下： 频率等级 A级 ≤±0.05Hz B级 ≤±0.5Hz C级 ≤±1Hz
xPredict = np.array([voltage, current, frequence]).reshape(-1, 3) # 这里将x转换为3维数组
yPredict = mlr.predict(xPredict)

print ("5.预测数据为: ",xPredict)
print ("6.预测结果为:",yPredict)
