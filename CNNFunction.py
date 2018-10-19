#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:23:16 2018

@author: jinyx
"""
import tensorflow as tf
# 构建网络
def buildCNN(w, h, c):
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    # 第一个卷积层 + 池化层
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=5,
        kernel_size=[1, 171],
        padding="same", #全零填充
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 5], strides=2)

    re1 = tf.reshape(pool1, [-1, 6 * 6 * 128])
    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense1,
                             units=2,  
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    return logits, x, y_

# 返回损失函数的值，准确值等参数
def accCNN(logits, y_):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, train_op, correct_prediction, acc


