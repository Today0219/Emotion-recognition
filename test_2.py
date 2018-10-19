# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:41:51 2018

@author: jinyu
"""
'''
import tensorflow as tf  
# 设计Graph  
x1 = tf.constant([2, 3, 4])  
x2 = tf.constant([4, 0, 1])  
x3 = tf.constant([1, 1, 1]) 
y = tf.add(x1, x2)  
# 打开一个session --> 计算y  
with tf.Session() as sess:  
    print(sess.run(y))  
'''
import tensorflow as tf  
# 设计Graph  
x1 = tf.placeholder(tf.int16)  
x2 = tf.placeholder(tf.int16)  
y = tf.add(x1, x2)  
# 用Python产生数据  
li1 = [2, 3, 4]  
li2 = [4, 0, 1]  
# 打开一个session --> 喂数据 --> 计算y  
with tf.Session() as sess:  
    print(sess.run(y, feed_dict={x1: li1, x2: li2})) 