import tensorflow as tf


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
