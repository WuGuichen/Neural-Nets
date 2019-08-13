import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 定义Wx_plus_b, 即神经网络未激活的值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 使用激励函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 构建所需数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义神经层: 输入层1个、隐藏层10个、输出层1个
# 搭建网络
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
# 对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
# 让机器学习提升它的准确率
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 使用变量前的初始化
init = tf.global_variables_initializer()

# 定义Session
sess = tf.Session()
sess.run(init)

# 开始训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
