# 1. pythonDataAnalyse
写在前面：前段时间，由于tutor问我们要不要去投稿论文，当时想着最近时间也还算挺多，考研方面的事情也还没有正式开始，因此就想着去尝试一下。然后项目的背景就是2022-2023年的大创项目，这个项目其实总体来说还算是简单，大致方案是利用神经网络系统对历史数据进行清洗、分析、训练，最终形成我们想要的模型，当时我还和guo同学一起讨论过具体的方案，他的建议是就做个简单的线性网络就行，不用那么复杂，能跑就行，因此就有了我对这些项目的学习。当然，本人对python这方面也只是稍作涉猎，就浅尝辄止学习了一部分，实际上对于很多具体的内容并没有深入的了解，希望大家在分析我的项目时，多多提出宝贵的意见！
<p align='right'>——3/13/23</p>

## 1.1 Tensorflow预测模型

### 1.1.1 样例（html）
这部分首先是一个[1.TensorFlow_example.html](https://github.com/LoveQinxia/pythonDataAnalyse/blob/main/Tensorflow%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/1.TensorFlow_example.html "1.TensorFlow_example.html")样例，这个是我以前学习html的时候，在TensorFlow的官网中无意间找到的案例，当时觉得对自己有很大的启发，就把这个样例给保存下来了，看不懂语法没关系，重要是看看它的运行结果：

![TensorFlow预测结果.jpg](https://github.com/LoveQinxia/pythonDataAnalyse/blob/main/source/TensorFlow%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.jpg)#pic_center

### 1.1.2 TensorFlow线性回归预测模型
1. 首先我们导入必要的模块：
~~~
from __future__ import print_function, division
# import tensorflow as tf
import tensorflow.compat.v1 as tf # 使用TensorFlow1.x
tf.disable_v2_behavior() # 弃用TensorFlow2.x的行为

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
~~~

2. 然后开始读取数据（在这步之前，请先记得生成对应的数据文件，用[生成历史电网数据.py](https://github.com/LoveQinxia/pythonDataAnalyse/blob/main/Tensorflow%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/%E7%94%9F%E6%88%90%E5%8E%86%E5%8F%B2%E7%94%B5%E7%BD%91%E6%95%B0%E6%8D%AE.py "生成历史数据.py")进行生成对应的[电网数据.csv](https://github.com/LoveQinxia/pythonDataAnalyse/blob/main/Tensorflow%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/grid_data_TensorFlow.csv)，生产的结果如下图所示）：

![TensorFlow训练的20行数据.jpg](https://github.com/LoveQinxia/pythonDataAnalyse/blob/main/source/TensorFlow%E8%AE%AD%E7%BB%83%E7%9A%8420%E8%A1%8C%E6%95%B0%E6%8D%AE.jpg)
~~~
# 利用pandas库读取csv数据
train = pd.read_csv("grid_data_TensorFlow.csv")
~~~

3. 定义一些神经网络模型需要的基本量（样本samples、学习率learning rate、训练次数epochs）：
~~~
# 定义参数
train = train[train['voltage'] < 250] # 只会选取数值小于250的电压，+10%
train = train[train['voltage'] > 176] # 只会选取数值大于176的电压，-10%
train_X = train['voltage'].values.reshape(-1,1)
train_Y = train['current'].values.reshape(-1,1)
n_samples = train_X.shape[0]
# 设置学习率
learning_rate = 0.075 # 这个数值真的需要自己去探索,我从2、1、0.1、0.01、0.05开始尝试，这些不是偏大就是偏小，而0.075刚刚好，从数据分布集当中直接穿过去
# 设置训练次数
training_eppchs = 180 # 训练轮数
# 设置多少次显示一次
display_step = 20
# 定义X,Y占位符
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
~~~

4. 开始构建神经网络模型：
~~~
# 使用Variable定义学习参数
W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32) # weight权重
b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32) # bias偏置
# 构建正向传播结构
pred = tf.add(tf.multiply(W, X), b)
# 损失函数loss
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# 使用梯度下降优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 激活init
init = tf.global_variables_initializer()
~~~

5. 训练神经网络模型：
~~~
# 启动session,初始化变量
with tf.Session() as sess:
    sess.run(init)
# 启动循环开始训练
    for epoch in range(training_eppchs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer,feed_dict={X:x, Y:y})
# 显示训练中的详细信息
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.3f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b),'\n')
~~~
神经网络训练过程如图所示：

![TensorFlow训练过程截图.jpg](https://github.com/LoveQinxia/pythonDataAnalyse/blob/main/source/TensorFlow%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E6%88%AA%E5%9B%BE.jpg)

6. 结果展示（matplotlib中的函数）：
~~~
# 展示训练结果
    plt.plot(train_X, train_Y, 'ro', label="Original data") # 设置图像的需要的参数
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show() # 展示图形代码，没有这一句就无法弹出图像框
~~~
最终结果如下图所示（参数的设置请看***定义参数***）：

![训练结果.png](https://github.com/LoveQinxia/pythonDataAnalyse/blob/main/source/1-1%E3%80%81200-0.1-180.png)

7. （补充）生成历史数据：
~~~
import pandas as pd
import numpy as np

# 生成电压、电流的数据，分别总共生成了200个
voltage = np.random.normal(loc=220, scale=11/3, size=200) # 正态分布函数normal()，均值loc=220，标准差scale=11/3
current = np.random.normal(loc=5, scale=0.5/3, size=200) # 正态分布函数normal()，均值loc=5，标准差scale=0.5/3

# 将数据合并为数据框
data = pd.DataFrame({'voltage': voltage, 'current': current})
# 看看它是什么成分
print(type(data))
# 将数据保存为CSV文件
data.to_csv('grid_data_TensorFlow.csv', index=False)

# 将数据保存为Excel文件
# data.to_excel('grid_data.xlsx', index=False)
~~~

解释下normal函数：
>正态分布函数，normal(loc, scale, size)，loc是样本均值、scale是样本标准差(3sigema原则：包含99%以上的分布，size是选取的样本)


## 2.1多元线性回归拟合电网数据
1. 生产历史数据
   1. 导入必要的库函数
   ~~~
    import pandas as pd
    import numpy as np
   ~~~
   2. 定义基本的函数变量。**start_time**：表示了开始时间，**end_time**：表示了结束时间。**timestamp**：截取时间范围，变化范围为1min
   ~~~
   # 生成时间戳
    start_time = '2022-01-01 00:00:00'
    end_time = '2022-01-01 23:59:59'
    timestamp = pd.date_range(start=start_time, end=end_time, freq='1min')
   ~~~
    3. 生成对应的电网数据
    ~~~
    # 生成电压、电流、频率、功率的数据
    voltage = np.random.normal(loc=220, scale=11/3, size=len(timestamp)) # 正态分布函数normal()，均值loc=220，标准差scale=11/3
    current = np.random.normal(loc=5, scale=0.5/3, size=len(timestamp)) # 正态分布函数normal()，均值loc=5，标准差scale=0.5/3
    frequency = np.random.normal(loc=50, scale=0.05, size=len(timestamp)) # 正态分布函数normal()，均值loc=50，标准差scale=0.05
    power = voltage * current # 功率 = 电压 * 电流
    ~~~
    4. 合并数据以及导出为CSV文件
    ~~~
    # 将数据合并为数据框
    data = pd.DataFrame({'timestamp': timestamp, 'voltage': voltage, 'current': current, 'frequency': frequency, 'power': power})
    print(type(data))
    # 将数据保存为CSV文件
    data.to_csv('grid_data.csv', index=False)
    ~~~

2. 多元线性回归拟合电网数据
   1. 导入必要的库函数
   ~~~
    from numpy import genfromtxt
    import numpy as np
    from sklearn import linear_model
    import pandas as pd
   ~~~
   2. 读取数据文件数据
   ~~~
    datapath = r"grid_data.csv" # 读取数据文件
    data = genfromtxt(datapath,delimiter=",") # genfromtxt() 是 NumPy 中的一个函数，它可以用来读取 CSV 文件（逗号分隔值文件）或其他文本文件，并将其转换为 NumPy 数组。下面是 genfromtxt() 的基本用法：

    x = data[1:,:-1] # 你看这里，起步是1，之前那个是[:,:-1]，因此可以推断，这里起步的1就是意味着从第1行也就是真实的第二行开始读取
    y = data[1:,-1]
    print ("1.输出数据集x:\n",x,"\n") # 输出数据集x
    print ("2.输出数据集y:\n",y,"\n") # 输出数据集y
   ~~~
   3. 定义线性回归模型
   ~~~

    mlr = linear_model.LinearRegression() # 定义线性回归函数，我不在乎你是不是多元，反正我先定义

    mlr.fit(x, y) # 拟合变量x，y
    print("3.系数:",mlr.coef_,"\n") # 打印系数
    print("4.截距:",mlr.intercept_,"\n") # 打印截距，我估计默认是纵轴截距
   ~~~
   4. 输入预测数据和输出预测结果
    ~~~
    # 输入预测数据集（单次预测）
    voltage = np.random.normal(220, 11) # 电压的波动在±5%，220*0.05=11
    current = np.random.normal(5,0.5) # 电流的波动在±10%，5*0.1=0.5
    frequency = np.random.normal(50,0.05) # 国标波动等级规定如下： 频率等级 A级 ≤±0.05Hz B级 ≤±0.5Hz C级 ≤±1Hz
    xPredict = np.array([voltage, current, frequency]).reshape(-1, 3) # 这里将x转换为3维数组
    yPredict = mlr.predict(xPredict)
    # 输出预测结果（单次预测）
    print ("5.预测数据为: ",xPredict,type(xPredict))
    print ("6.预测结果为:",yPredict,type(yPredict),"\n")
    ~~~
   


[超链接显示名](超链接地址 "超链接title")
![图片alt](图片链接 "图片title")。
