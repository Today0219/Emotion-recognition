# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:53:45 2018
直接对4个象限的情绪结果做预测，因为皮肤电对唤醒度的预测没啥效果
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

###############################################################################
if False: #计算四个象限的情绪
    print("把连续的唤醒度和愉悦度转化为离散的二分类值（4个象限对应四种情绪）")
    print("---------happy emotion----------")
    df_result = all_df_y
    a = df_result[df_result.valence>=5].index
    b = df_result[df_result.arousal>=5].index
    happy_index = [val for val in a if val in b]
    print("len(happy_index)=",len(happy_index)) 
    df_result['happy'] = -1
    for i in happy_index:
        df_result['happy'].loc[i] = 1
    print("---------sad emotion----------")
    df_result = all_df_y
    a = df_result[df_result.valence<=5].index
    b = df_result[df_result.arousal<=5].index
    sad_index = [val for val in a if val in b]
    print("len(sad_index)=",len(sad_index)) 
    df_result['sad'] = -1
    for i in sad_index:
        df_result['sad'].loc[i] = 1
    print("---------nervous emotion----------")
    df_result = all_df_y
    a = df_result[df_result.valence<5].index
    b = df_result[df_result.arousal>5].index
    nervous_index = [val for val in a if val in b]
    print("len(nervous_index)=",len(nervous_index)) 
    df_result['nervous'] = -1
    for i in nervous_index:
        df_result['nervous'].loc[i] = 1
    print("---------calm emotion----------")
    df_result = all_df_y
    a = df_result[df_result.valence>5].index
    b = df_result[df_result.arousal<5].index
    calm_index = [val for val in a if val in b]
    print("len(calm_index)=",len(calm_index)) 
    df_result['calm'] = -1
    for i in calm_index:
        df_result['calm'].loc[i] = 1
    print("四个情绪划分结果dump处理")
    pickle.dump(df_result,open("./dump_file/df_result","wb"))
else:
    print("读取四种情绪的dump文件")
    df_result = pickle.load(open("./dump_file/df_result","rb"))    
###############################################################################
def count_accuracy(ser1,ser2):
    sum_all = len(ser1)
    tmp = ser1==ser2
    sum_acc= len(tmp[tmp==True])
    return sum_acc/sum_all
if True:
    print("----------------‘happy’情绪预测----------------")
    data = GSR_feature_df
    target = df_result[['happy']]       
    print("######xgboost model CV######")
    for xgb_rounds in [50]:  
        xgb_model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,
                                      objective='binary:logistic',booster='gbtree',n_jobs=-1,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.5, reg_lambda=1.0,gamma=0,
                                      scale_pos_weight=1)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='accuracy')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
        xgb_pred_happy = cross_val_predict(xgb_model,data,target,cv=5)
        happy_acc = accuracy_score(xgb_pred_happy,df_result['happy'])
        print("happy_acc:",happy_acc)

if True:
    print("----------------‘sad’情绪预测----------------")
    data = GSR_feature_df
    target = df_result[['sad']]       
    print("######xgboost model CV######")
    for xgb_rounds in [50]:  
        xgb_model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,
                                      objective='binary:logistic',booster='gbtree',n_jobs=-1,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.5, reg_lambda=1.0,gamma=0,
                                      scale_pos_weight=1)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='accuracy')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
        xgb_pred_sad = cross_val_predict(xgb_model,data,target,cv=5)
        sad_acc = accuracy_score(xgb_pred_sad,df_result['sad'])
        print("sad_acc:",sad_acc)      

if True:
    print("----------------‘nervous’情绪预测----------------")
    data = GSR_feature_df
    target = df_result[['nervous']]       
    print("######xgboost model CV######")
    for xgb_rounds in [50]:  
        xgb_model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,
                                      objective='binary:logistic',booster='gbtree',n_jobs=-1,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.5, reg_lambda=1.0,gamma=0,
                                      scale_pos_weight=1)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='accuracy')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
        xgb_pred_nervous = cross_val_predict(xgb_model,data,target,cv=5)
        nervous_acc = accuracy_score(xgb_pred_nervous,df_result['nervous'])
        print("nervous_acc:",nervous_acc)

if True:
    print("----------------‘calm’情绪预测----------------")
    data = GSR_feature_df
    target = df_result[['calm']]       
    print("######xgboost model CV######")
    for xgb_rounds in [50]:  
        xgb_model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,
                                      objective='binary:logistic',booster='gbtree',n_jobs=-1,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.5, reg_lambda=1.0,gamma=0,
                                      scale_pos_weight=1)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='accuracy')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
        xgb_pred_calm = cross_val_predict(xgb_model,data,target,cv=5)
        calm_acc = accuracy_score(xgb_pred_calm,df_result['calm'])
        print("calm_acc:",calm_acc)

#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)  