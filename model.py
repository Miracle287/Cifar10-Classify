#!/usr/bin/python
# -*- coding: UTF-8 -*-
# CNN Structure (base on tensorflow)


import tensorflow as tf

# Network Parameters
n_classes = 10
dropout = 0.5


# 初始化权重：通过L2正则化的方法，减少特征权重，防止过拟合
def variable_with_weight_loss(shape, stddev, w1):
    # shape: 输入的卷积核大小，以及输入图像深度和输出深度（卷积核的个数）
    # stddev: 正态分布的标准差
    # w1: 控制是否L2正则化
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    # 如果w1存在，使用w1参数控制L2 Loss的大小
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return var


# 卷积层
def conv2D(x, weight):
    # x: 输入的tensor[batch_size, height, width, depth]
    # weight: 卷积核权重参数，对输入数据x做卷积操作
    # stride[0]=stride[3]=1, stride[1]为横向步长，stride[2]为纵向步长
    # padding: 池化操作, same-padding为外部补0，valid-padding控制在有效范围之内
    out = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')
    # out = tf.nn.bias_add(out, b)
    # out_channels = weight.get_shape()[3].value
    return out


# 池化层
def max_pool(x):
    # ksize[1, x, y, 1]
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")


# 激励层，使用线性修正单元
def relu(out, bias):
    # out: 卷积或者全连层输出结果
    # bias: 偏置
    return tf.nn.relu(tf.nn.bias_add(out, bias))


def dropout(d, value):
    return tf.nn.dropout(d, value)


def batch_normalization(sound, depth, training, scope='bn'):
    out_channels = depth

    with tf.variable_scope(scope):
        # Solve on Kennel
        batch_mean, batch_variance = tf.nn.moments(sound, [0, 1, 2], keep_dims=False)

        # Set Gamma and Beta
        gamma = tf.Variable(tf.ones([out_channels]), trainable=True)
        beta = tf.Variable(tf.zeros([out_channels]), trainable=True)

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, var = tf.cond(pred=tf.cast(training, tf.bool),
                            fn1=mean_var_with_update,
                            fn2=lambda: (ema.average(batch_mean), ema.average(batch_variance)))

    return tf.nn.batch_normalization(sound, mean, var, beta, gamma, variance_epsilon=0.001)


def conv_net(x, batch_size):
    # 第一层卷积层
    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
    conv1 = conv2D(x, weight1)
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
    relu1 = relu(conv1, bias1)
    pool1 = max_pool(relu1)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # 第二层卷积层
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
    conv2 = conv2D(norm1, weight2)
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    relu2 = relu(conv2, bias2)
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = max_pool(norm2)

    # 拉直数据, 变成一维以便进入全连接层
    reshape_x = tf.reshape(pool2, [batch_size, -1])  # 将batch_size张图片拉直
    dim_x = reshape_x.get_shape()[1].value  # 获取第二维度, 上一层输出的神经元个数

    # 第三层全连接层
    weight3 = variable_with_weight_loss([dim_x, 384], stddev=0.04, w1=0.004)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    fc_3 = relu(tf.matmul(reshape_x, weight3), bias3)  # 通过矩阵乘法，并通过激活函数，得到全连接输出的384维向量

    # 第四层全连接层
    weight4 = variable_with_weight_loss([384, 192], stddev=0.04, w1=0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    fc_4 = relu(tf.matmul(fc_3, weight4), bias4)

    # 第五层全连接层
    # 注意：这里输出10个单元，代表十个不同种类的图片特征向量，输出的并没有经过softmax
    weight5 = variable_with_weight_loss([192, 10], stddev=1 / 192, w1=0.0)
    bias5 = tf.Variable(tf.constant(0.1, shape=[10]))
    result = relu(tf.matmul(fc_4, weight5), bias5)

    # 经过softmax
    # result = tf.nn.softmax(result)
    return result