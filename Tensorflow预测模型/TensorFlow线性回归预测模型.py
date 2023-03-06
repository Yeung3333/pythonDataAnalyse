# coding:utf-8

from __future__ import print_function, division
# import tensorflow as tf
import tensorflow.compat.v1 as tf # 使用TensorFlow1.x
tf.disable_v2_behavior() # 弃用TensorFlow2.x的行为

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
train = pd.read_csv("grid_data_TensorFlow.csv")
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
# 展示训练结果
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()
