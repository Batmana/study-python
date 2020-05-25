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

im = mnist.train.images[1]
im = im.reshape(-1, 28)
pylab.imshow(im)
pylab.show()


