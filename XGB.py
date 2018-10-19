# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 02:57:14 2018
代码功能：XGB回归预测唤醒度和愉悦度，并根据结果组合预测4个象限的情绪
@author: jinyu
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
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

##############################下面用交叉验证做愉悦度预测########################
if True:
    print("----------------这是愉悦度预测----------------")
    data = GSR_feature_df
    target = all_df_y[['valence']]       
    print("######xgboost model CV######")
    for xgb_rounds in [50]:  
        xgb_model  = xgb.XGBRegressor(max_depth=5,learning_rate=0.1,n_estimators=xgb_rounds,
                                      objective='reg:linear',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                       reg_alpha=0.1, reg_lambda=0.8,gamma=1.0)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='neg_mean_absolute_error')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
        xgb_pred_valence = cross_val_predict(xgb_model,data,target,cv=5)
        
#############################下面用唤醒度做回归#################################
if True:
    print("----------------这是唤醒度预测----------------")
    data = GSR_feature_df
    target = all_df_y[['arousal']]   
    print("######xgboost regression model CV######")
    for xgb_rounds in [50]:  
        xgb_model  = xgb.XGBRegressor(max_depth=7,learning_rate=0.1,n_estimators=xgb_rounds,
                                      objective='reg:linear',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.90, colsample_bylevel=0.90,
                                       reg_alpha=0.1, reg_lambda=0.5,gamma=0)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='neg_mean_absolute_error')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
        xgb_pred_arousal = cross_val_predict(xgb_model,data,target,cv=5)

print("根据回归预测值构造4个象限的情绪2分类模型") 
df_v = pd.DataFrame(xgb_pred_valence,columns=['pred_v'],index=all_df_y.index)
df_a = pd.DataFrame(xgb_pred_arousal,columns=['pred_a'],index=all_df_y.index)
df_true_v = all_df_y[['valence']]
df_true_a = all_df_y[['arousal']]          
df_result = pd.concat([df_v,df_a,df_true_v,df_true_a],axis=1)

def count_accuracy(ser1,ser2):
    sum_all = len(ser1)
    tmp = ser1==ser2
    sum_acc= len(tmp[tmp==True])
    return sum_acc/sum_all
       
print("---------happy emotion----------")
happy_index = df_result[df_result.valence>=5].index.append(df_result[df_result.arousal>=5].index)
happy_index = set(happy_index)
print("len(happy_index)=",len(happy_index)) 
df_result['happy'] = -1
for i in happy_index:
    df_result['happy'].loc[i] = 1  
pred_happy_index = df_result[df_result.pred_v>=5].index.append(df_result[df_result.pred_a>=5].index)
pred_happy_index = set(pred_happy_index)
print("len(pred_happy_index)=",len(pred_happy_index)) 
df_result['pred_happy'] = -1
for i in pred_happy_index:
    df_result['pred_happy'].loc[i] = 1
acc = count_accuracy(df_result['pred_happy'],df_result['happy']) 
print("happy acc:",acc)

print("---------sad emotion----------")
sad_index = df_result[df_result.valence<5].index.append(df_result[df_result.arousal<5].index)
sad_index = set(sad_index)
print("len(sad_index)=",len(sad_index)) 
df_result['sad'] = -1
for i in sad_index:
    df_result['sad'].loc[i] = 1  
pred_sad_index = df_result[df_result.pred_v<5].index.append(df_result[df_result.pred_a<5].index)
pred_sad_index = set(pred_sad_index)
print("len(pred_sad_index)=",len(pred_sad_index)) 
df_result['pred_sad'] = -1
for i in pred_sad_index:
    df_result['pred_sad'].loc[i] = 1
acc = count_accuracy(df_result['pred_sad'],df_result['sad']) 
print("sad acc:",acc)


#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        