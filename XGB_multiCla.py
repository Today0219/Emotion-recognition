# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:31:09 2018
情绪的多分类问题
@author: jinyu
"""

import pandas as pd
import numpy as np
import pickle
import random
import xgboost as xgb
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
GSR_feature_df = pickle.load(open("./dump_file/df_feat_selected","rb"))
all_df_y = pickle.load(open("./dump_file/all_df_y","rb"))
print("GSR_feature_df.shape:",GSR_feature_df.shape)

print("数据缩放处理，归一化处理")
min_max_scaler = MinMaxScaler()
GSR_feature_df = min_max_scaler.fit_transform(GSR_feature_df)

print("把连续的唤醒度和愉悦度转化为离散的4个类别值")
print("---------happy emotion----------")
df_result = all_df_y
a = df_result[df_result.valence>=5].index
b = df_result[df_result.arousal>=5].index
#happy_index = [val for val in a if val in b]
happy_index = set(a).intersection(set(b))
print("len(happy_index)=",len(happy_index)) 
df_result['4emotion'] = -1
for i in happy_index:
    df_result['4emotion'].loc[i] = 0
print("---------sad emotion----------")
df_result = all_df_y
a = df_result[df_result.valence<=5].index
b = df_result[df_result.arousal<=5].index
#sad_index = [val for val in a if val in b]
sad_index = set(a).intersection(set(b))
print("len(sad_index)=",len(sad_index)) 
for i in sad_index:
    df_result['4emotion'].loc[i] = 1
print("---------nervous emotion----------")
df_result = all_df_y
a = df_result[df_result.valence<5].index
b = df_result[df_result.arousal>5].index
#nervous_index = [val for val in a if val in b]
nervous_index = set(a).intersection(set(b))
print("len(nervous_index)=",len(nervous_index)) 
for i in nervous_index:
    df_result['4emotion'].loc[i] = 2
print("---------calm emotion----------")
df_result = all_df_y
a = df_result[df_result.valence>5].index
b = df_result[df_result.arousal<5].index
#calm_index = [val for val in a if val in b]
calm_index = set(a).intersection(set(b))
print("len(calm_index)=",len(calm_index)) 
for i in calm_index:
    df_result['4emotion'].loc[i] = 3
    
###############################################################################     
if True:
    print("训练多分类器") 
    data = GSR_feature_df
    target = all_df_y[['4emotion']]       
    print("######xgboost model CV######")
    for xgb_rounds in [50]:  
        xgb_model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,
                                      objective='multi:softmax',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.5, reg_lambda=1.0,gamma=0,
                                      scale_pos_weight=1)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='accuracy')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
        xgb_pred_4emotion = cross_val_predict(xgb_model,data,target,cv=5)
        acc_4emotion = accuracy_score(xgb_pred_4emotion,df_result['4emotion'])
        print("4emotion_acc:",acc_4emotion)   
    
#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)     
    
    
    
    
    