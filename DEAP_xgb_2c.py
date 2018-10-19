# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:04:47 2018

@author: jinyx
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#读取原始特征
df_feat = pickle.load(open("./dump_file/df_feat_selected","rb"))
#读取标签（Y值）
all_df_y_2c = pickle.load(open("./dump_file/all_df_y_2c","rb"))
print("df_feat.shape:",df_feat.shape)

train_X,test_X,train_Y,test_Y = \
    train_test_split(df_feat,all_df_y_2c,test_size=0.2,random_state=1000)

print("train_X.shape:",train_X.shape)
print("test_X.shape:",test_X.shape)  
 
# 
dtrain = xgb.DMatrix(train_X, train_Y)
dtest = xgb.DMatrix(test_X,test_Y)
    
xgb_params = {
    'booster': 'gbtree',
    
    'colsample_bytree': 0.8,
    
    'colsample_bylevel': 0.8,

    'eta': 0.01,

    'max_depth': 6,

    'objective': 'binary:logistic',

    'eval_metric': 'error',

    'silent':0,
}

watchlist = [(dtrain, 'train'), (dtest, 'test')]
num_round = 300
bst = xgb.train(xgb_params, dtrain, num_round,evals=watchlist)

y_pred = bst.predict(dtest)

df_y_pred = pd.DataFrame(y_pred,columns=['temp_pred_y'])
df_y_pred['pred_y'] = 0
df_y_pred['pred_y'][df_y_pred['temp_pred_y'] >= 0.5] = 1 
df_y_pred['pred_y'][df_y_pred['temp_pred_y'] < 0.5] = 0     
print(accuracy_score(test_Y, df_y_pred['pred_y']))



                                          