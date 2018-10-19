# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
#用来计算程序运行时间
import datetime
starttime = datetime.datetime.now()

#读取数据
GSR_feature_df = pickle.load(open("./dump_file/df_feat_selected","rb"))
all_df_y_valence = pickle.load(open("./dump_file/all_df_y_valence","rb"))
all_df_y = pickle.load(open("./dump_file/all_df_y","rb"))
all_df_y_2c = pickle.load(open("./dump_file/all_df_y_2c","rb"))
print("GSR_feature_df.shape:",GSR_feature_df.shape)

print("数据缩放处理，归一化处理")
min_max_scaler = MinMaxScaler()
GSR_feature_df = min_max_scaler.fit_transform(GSR_feature_df)

##############################下面用交叉验证做愉悦度预测##################################
if False:
    print("----------------这是愉悦度预测----------------")
    data = GSR_feature_df
    target = all_df_y_valence
    
    print("######linear regression CV######")
    linearR_model = LinearRegression()
    linearR_scores = cross_val_score(linearR_model,data,target,cv=5,scoring='neg_mean_absolute_error')
    print("linearR_scores:",abs(linearR_scores))
    print("linearR_scores_mean:",abs(linearR_scores.mean()))
    
    print("######ridge model CV######")
    for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
        ridge_model = Ridge(alpha=alpha)
        ridge_scores = cross_val_score(ridge_model,data,target,cv=5,scoring='neg_mean_absolute_error')
        print("alpha:%.1f->ridge_scores_mean:%f"%(alpha,abs(ridge_scores.mean())))
    
    print("######SVR model CV######")
    for c in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,4.0]:
        svr_model =  SVR(C=c,kernel='rbf')
        svr_scores = cross_val_score(svr_model,data,target,cv=5,scoring='neg_mean_absolute_error')   
        print("c:%.1f->svr_scores_mean:%f"%(c,abs(svr_scores.mean())))
        
    print("######xgboost model CV######")
    for xgb_rounds in [20,30,40,50,60,70]:  
        xgb_model  = xgb.XGBRegressor(max_depth=5,learning_rate=0.1,n_estimators=xgb_rounds,
                                      objective='reg:linear',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                       reg_alpha=0.1, reg_lambda=0.8,gamma=1.0)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='neg_mean_absolute_error')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
    
    print("######xgboost classification model CV######")
    target = all_df_y_2c['emotion_2']
    for xgb_rounds in [40,50,60,70]:  
        xgb_model  = xgb.XGBClassifier(max_depth=7,learning_rate=0.1,n_estimators=xgb_rounds,
                                      objective='binary:logistic',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.90, colsample_bylevel=0.90,
                                       reg_alpha=0.1, reg_lambda=0.5,gamma=0)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='accuracy')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))

#############################下面用唤醒度做回归#################################
if True:
    print("----------------这是唤醒度预测----------------")
    data = GSR_feature_df
    target = all_df_y[['arousal']]
    '''    
    print("######linear regression CV######")
    linearR_model = LinearRegression()
    linearR_scores = cross_val_score(linearR_model,data,target,cv=5,scoring='neg_mean_absolute_error')
    print("linearR_scores:",abs(linearR_scores))
    print("linearR_scores_mean:",abs(linearR_scores.mean()))
    
    print("######ridge model CV######")
    for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
        ridge_model = Ridge(alpha=alpha)
        ridge_scores = cross_val_score(ridge_model,data,target,cv=5,scoring='neg_mean_absolute_error')
        print("alpha:%.1f->ridge_scores_mean:%f"%(alpha,abs(ridge_scores.mean())))
        
    print("######SVR model CV######")
    for c in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
        svr_model =  SVR(C=c,kernel='rbf')
        svr_scores = cross_val_score(svr_model,data,target,cv=5,scoring='neg_mean_absolute_error')   
        print("c:%.1f->svr_scores_mean:%f"%(c,abs(svr_scores.mean())))
    ''' 
    print("######xgboost regression model CV######")
    for xgb_rounds in [40,50,60,70]:  
        xgb_model  = xgb.XGBRegressor(max_depth=7,learning_rate=0.1,n_estimators=xgb_rounds,
                                      objective='reg:linear',booster='gblinear',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.90, colsample_bylevel=0.90,
                                       reg_alpha=0.1, reg_lambda=0.5,gamma=0)
        xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='neg_mean_absolute_error')
        print("xgb_rounds:%d->xgb_scores_mean:%f"%(xgb_rounds,abs(xgb_scores.mean())))
    
    '''
    print("######KNN regression model CV######")
    for knn_neighbors in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
        knn_model = KNeighborsRegressor(n_neighbors=knn_neighbors)
        knn_scores = cross_val_score(knn_model,data,target,cv=5,scoring='neg_mean_absolute_error')
        print("knn_neighbors:%d->knn_scores_mean:%f"%(knn_neighbors,abs(knn_scores.mean())))
           
    print("######MLP regression model CV######")
    for mlp_alpha in [0.1,0.01,0.001,0.0001]:
        mlp_model = MLPRegressor(hidden_layer_sizes=(1000, ),alpha=mlp_alpha)
        mlp_scores = cross_val_score(mlp_model,data,target,cv=5,scoring='neg_mean_absolute_error',n_jobs=1)
        print("mlp_alpha:%f->mlp_scores_mean:%f"%(mlp_alpha,abs(mlp_scores.mean())))
    '''
    
#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)
























