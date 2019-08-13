import numpy as np
import tensorflow as tf

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# 搭建模型
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biase = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biase

# 计算误差
loss = tf.reduce_mean(tf.square(y-y_data))

# 反向误差传递：梯度下降法：Gradient Descent
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化所有之前定义的Variable
init = tf.compat.v1.global_variables_initializer()

# 创建会话Session
sess = tf.compat.v1.Session()
sess.run(init)       # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biase))
