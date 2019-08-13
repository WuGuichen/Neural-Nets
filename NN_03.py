import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# MNIST库是手写体数字库:55000张训练图片，每张图片的分辨率是28×28
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 搭建网络
xs = tf.placeholder(tf.float32, [None, 784])     # 28*28
ys = tf.placeholder(tf.float32, [None, 10])      # 0-9
# add out put layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# Cross entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  #loss
# 用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# train start
# 训练集
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

