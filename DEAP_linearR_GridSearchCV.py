# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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
print("GSR_feature_df.shape:",GSR_feature_df.shape)

print("数据缩放处理，归一化处理")
min_max_scaler = MinMaxScaler()
GSR_feature_df = min_max_scaler.fit_transform(GSR_feature_df)

##############################下面用交叉验证做##################################
print("----------------这是愉悦度预测----------------")
data = GSR_feature_df
target = all_df_y_valence
print("######linear regression CV######")
linearR_model = LinearRegression()
linearR_scores = cross_val_score(linearR_model,data,target,cv=5,scoring='neg_mean_absolute_error')
print("linearR_scores:",abs(linearR_scores))
print("linearR_scores_mean:",abs(linearR_scores.mean()))

print("######lasso model CV######")
param_grid = {'alpha':[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]}
lasso_model = Lasso()
gsearch = GridSearchCV(lasso_model,param_grid,cv=5,scoring='neg_mean_absolute_error')
gsearch.fit(data,target)
print("lasso->best_params:",gsearch.best_score_)

print("######ridge model CV######")
param_grid = {'alpha':[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]}
Ridge_model = Ridge()
gsearch = GridSearchCV(Ridge_model,param_grid,cv=5,scoring='neg_mean_absolute_error')
gsearch.fit(data,target)
print("Ridge->best_params:",gsearch.best_score_)

print("######xgboost model CV######")
param_grid = {'max_depth':[3],
              'learning_rate':[0.1],
              'n_estimators':[50],
              'objective':['reg:linear'],
              'booster':['gbtree'],
              'n_jobs':[10],
              'subsample':[1],
              'colsample_bytree':[1.0],
              'colsample_bylevel':[1.0],
              'reg_alpha':[1.0],
              'reg_lambda':[1.0],
              'gamma':[1.0],
        }
xgb_model  = xgb.XGBRegressor()
gsearch = GridSearchCV(xgb_model,param_grid,cv=5,scoring='neg_mean_absolute_error',n_jobs=10)
gsearch.fit(data,target)
print("Ridge->best_params:",gsearch.best_score_)
#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)
























