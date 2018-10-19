# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import MinMaxScaler

GSR_selected_feature_df = pickle.load(open("./dump_file/GSR_selected_feature_df","rb"))
all_df_y_mutiLable = pickle.load(open("./dump_file/all_df_y_mutiLable","rb"))
print('GSR_selected_feature_df.shape:',GSR_selected_feature_df.shape)
print('all_df_y_mutiLable.shape:',all_df_y_mutiLable.shape)

scaler = MinMaxScaler()
scaler.fit(GSR_selected_feature_df)
data = scaler.transform(GSR_selected_feature_df)
data_df = pd.DataFrame(data)

train_X = data_df.iloc[:int(1280*0.7), :].values
test_X = data_df.iloc[int(1280*0.7):,:].values

train_Y = all_df_y_mutiLable.iloc[:int(1280*0.7), :].values
test_Y = all_df_y_mutiLable.iloc[int(1280*0.7):,:].values

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 8
param['num_class'] = 4

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))



