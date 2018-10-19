#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:10:21 2018
CNN做分类
@author: jinyx
"""
import pandas as pd
import numpy as np
import pickle
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
#用来计算程序运行时间
import datetime
starttime = datetime.datetime.now()
#读取数据
for eeg_CH in range(1,33,1):
    file_path = "./dump_file/CH{}_df_EEG_x".format(eeg_CH)
    df_data = pickle.load(open(file_path,"rb"))
    locals()["CH{}_df_EEG_x".format(eeg_CH)] = df_data
all_df_y = pickle.load(open("./dump_file/all_df_y","rb"))
for i in range(0,1280,1): #总共1280个实验所以会有1280个二维矩阵
    locals()["mat{}".format(i)]=pd.DataFrame()
    for eeg_CH in range(1,33,1): #脑电共有32个通道，所以一个矩阵大小32*8064
        locals()["mat{}".format(i)] = locals()["mat{}".format(i)].\
            append(locals()["CH{}_df_EEG_x".format(eeg_CH)].iloc[i:i+1],ignore_index=True)

#
INPUT_NODE = 258048 #32*8064
OUTPUT_NODE = 2 #2分类

#先简单的用1000个作为训练集，280个作为测试集