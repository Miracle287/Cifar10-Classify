# -*- coding:utf-8 -*-

import tensorflow as tf
import inputs
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import model

# 每个批次的大小, batch_size 应大于输入长度
batch_size = 128


# 标准化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    # return data / 255


def get_batch(data, label, batch_size, step):
    start = (step * batch_size) % len(data)
    end = (step * batch_size + batch_size) % len(data)

    if start > end:
        start = end
        end = start + batch_size
    elif start == end:
        start = 0
        end = start + batch_size
    return normalization(data[start:end]), label[start:end]


# 计算损失函数
def loss_func(logits, labels):
    # 交叉熵损失计算
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # # 加入到之前的正则化损失
    tf.add_to_collection("losses", cross_entropy_mean)
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


# 交叉熵损失函数计算精确度
def evaluation_accuracy(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        correct = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct)
    return accuracy


# 可视化曲线
def show_history(history):
    # show figure
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim([0, 5])
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(122)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='lower left')
    plt.show()


def display_step_train_period(step, loss, accuracy, val_loss, val_accuracy, pic_per_sec, sec_per_batch):
    print("step: %d, loss=%.3f, accuracy=%.3f, val_loss=%.3f, val_accuracy=%.3f, %.1f pictures/sec, %.3f sec/batch"
          % (step, loss, accuracy, val_loss, val_accuracy, pic_per_sec, sec_per_batch))


# 保存模型
def save_model(file):
    saver = tf.train.Saver()
    # variables to disk.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # Save the variables to disk.
        save_path = saver.save(sess, file)
        print("Model saved in file: ", save_path)


# 恢复模型
def load_model(file):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, file)
        print("Model restored.")


if __name__ == '__main__':
    # 用于评估验证集和测试集
    def evaluate(sess, x_, y_, batch_size):
        data_len = len(x_)
        batch_eval = int(data_len / batch_size)
        total_loss = 0.0
        total_acc = 0.0
        for step in range(batch_eval):
            x_batch, y_batch = get_batch(x_, y_, batch_size, step)
            loss, acc = sess.run([losses, accuracy], feed_dict={x: x_batch, y: y_batch})
            total_loss += loss * batch_size
            total_acc += acc * batch_size
        return total_loss / data_len, total_acc / data_len

    # 创建placeholder, 用于提供输入数据
    x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    y = tf.placeholder(tf.int32, [batch_size, 10])

    # 训练网络
    result = model.conv_net(x, batch_size)
    # result = cifar10_model.network(x)

    # 计算损失函数
    losses = loss_func(result, y)
    # 设置优化算法使得成本最小, 全局学习率为0.0001
    train_op = tf.train.AdamOptimizer(0.0001).minimize(losses)
    # 获取最高分类准确率，以top1为标准
    # top_k_op = tf.nn.in_top_k(result, y, 1)
    # 评估模型准确率
    accuracy = evaluation_accuracy(result, y)

    train_x = inputs.load(os.path.join(inputs.input_train_dir, "data.dat"))
    train_y = inputs.load(os.path.join(inputs.input_train_dir, "label.dat"))
    test_x = inputs.load(os.path.join(inputs.input_test_dir, "data.dat"))
    test_y = inputs.load(os.path.join(inputs.input_test_dir, "label.dat"))

    # 打乱训练集
    train_x, train_y = shuffle(train_x, train_y)

    # 生成验证集
    validation_size = int(len(train_x) * 0.1)
    validation_x, validation_y = train_x[:validation_size], train_y[:validation_size]
    train_x, train_y = train_x[validation_size:], train_y[validation_size:]

    # 开始训练
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()

        import time
        display_history = {
            'loss': [],
            'acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # 最大训练步骤
        max_step = 60000

        for step in range(max_step):
            image_batch, label_batch = get_batch(train_x, train_y, batch_size, step)
            optimizer = sess.run(train_op, feed_dict={x: image_batch, y: label_batch})
            if step % 100 == 0:
                start_time = time.time()
                acc, loss = sess.run([accuracy, losses], feed_dict={x: image_batch, y: label_batch})
                duration = time.time() - start_time
                speed = batch_size / duration

                # Evaluate validation
                val_loss, val_acc = evaluate(sess, validation_x, validation_y, batch_size)

                # Append to history
                display_history['loss'].append(loss)
                display_history['acc'].append(acc)
                display_history['val_loss'].append(val_loss)
                display_history['val_acc'].append(val_acc)

                # 打印每轮训练的耗时
                display_step_train_period(step, loss, acc, val_loss, val_acc, speed, duration)

        # 输入测试集评估
        test_loss, test_acc = evaluate(sess, test_x, test_y, batch_size)
        print("test accuracy:%.3f" % test_acc)

        # 曲线保存
        inputs.save(os.path.join(inputs.output_dir, "train_history"), display_history)
        show_history(display_history)

        # 模型保存
        save_model(os.path.join(inputs.output_dir, "model.final"))










