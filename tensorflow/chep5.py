# -*- coding: UTF-8 -*-
"""
实现识别图中模糊的手写数字
date:2020/05/25
"""
# 1.下载数据集，
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('输入数据:', mnist.train.images)
print('输入数据shape', mnist.train.images.shape)
import pylab

# im = mnist.train.images[1]
# im = im.reshape(-1, 28)
# pylab.imshow(im)
# pylab.show()
#
# # 测试集
# print('输入数据shape', mnist.test.images.shape)
# # 验证集
# print('输入数据shape', mnist.validation.images.shape)

# 分析图谱特点，定义变量
import tensorflow as tf
tf.reset_default_graph()
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义输出节点
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义反向传播的结构
# print(pred)
# exit(1)
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate = 0.01

# 使用剃度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1

# 启动session
with tf.Session() as sess:
    # Initializtins OP
    sess.run(tf.global_variables_initializer())

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        # 循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})

            # 计算平均loss
            avg_cost += c / total_batch

        # 显示训练中的详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Finished")

# 测试Model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

    saver = tf.train.Saver()
    model_path = "log/521model.ckpt"
    save_path = saver.save(sess, model_path)

# 读取模型
print("Starting 2nd session")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 恢复模型变量
    saver.restore(sess, model_path)

    # 测试Model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # tf.argmax axis = 1 的时候返回每一行最大值的位置索引
    outpuut = tf.argmax(pred, 1)

    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([outpuut, pred], feed_dict={x: batch_xs})

    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
