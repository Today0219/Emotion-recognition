# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb 
import warnings
warnings.filterwarnings("ignore")
#用来计算程序运行时间
import datetime
starttime = datetime.datetime.now()

print("######读取数据（基于皮肤电）######")
GSR_feature_df = pickle.load(open("./dump_file/df_feat_selected","rb"))
all_df_y_valence = pickle.load(open("./dump_file/all_df_y_valence","rb"))
all_df_y = pickle.load(open("./dump_file/all_df_y","rb"))
all_df_y_2c = pickle.load(open("./dump_file/all_df_y_2c","rb"))
print("GSR_feature_df.shape:",GSR_feature_df.shape)

print("######数据缩放处理，归一化处理######")
min_max_scaler = MinMaxScaler()
GSR_feature_df = min_max_scaler.fit_transform(GSR_feature_df)

#############################下面用唤醒度做分类#################################
if True:
    print("----------------这是高低愉悦度度二分类预测----------------")
    data = GSR_feature_df
    target = all_df_y_2c #高低愉悦度
    
    #贝叶斯效果不好，可能是数据不服从高斯（正态）分布
    #print("######NB classification CV######") 
    #NB_model = GaussianNB()
    #NB_scores = cross_val_score(NB_model,data,target,cv=5,scoring='accuracy')
    #print("NB_scores:",abs(NB_scores))
    #print("NB_scores_mean:",abs(NB_scores.mean()))
    
    print("######KNN classification CV######")
    KNN_model = KNeighborsClassifier(n_neighbors=20)
    KNN_scores = cross_val_score(KNN_model,data,target,cv=5,scoring='accuracy')
    print("KNN_scores:",abs(KNN_scores))
    print("KNN_scores_mean:",abs(KNN_scores.mean()))
    
    print("######xgb classification CV######")
    xgb_model = xgb.XGBClassifier(max_depth=6,learning_rate=0.01,n_estimators=300,
                                      objective='binary:logistic',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.5, reg_lambda=1.0,gamma=0,
                                      scale_pos_weight=1)
    xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='accuracy')
    print("xgb_scores:",abs(xgb_scores))
    print("xgb_scores_mean:",abs(xgb_scores.mean()))
    
    print("######MLP classification CV######")
    mlp_model = MLPClassifier(hidden_layer_sizes=(500,2),alpha=0.1)
    mlp_scores = cross_val_score(mlp_model,data,target,cv=5,scoring='accuracy')
    print("mlp_scores:",abs(mlp_scores))
    print("mlp_scores_mean:",abs(mlp_scores.mean()))
    

#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)



