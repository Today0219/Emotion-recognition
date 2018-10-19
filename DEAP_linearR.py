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

#print("数据缩放处理，具有零均值和单位方差")
#stdScaler = preprocessing.StandardScaler()
#stdScaler.fit(GSR_feature_df)
#stdScaler.transform(GSR_feature_df)
#print("mean:\n{}".format(GSR_feature_df.mean(axis=0)))
#print("std:\n{}".format(GSR_feature_df.std(axis=0)))

print("数据缩放处理，归一化处理")
min_max_scaler = MinMaxScaler()
GSR_feature_df = min_max_scaler.fit_transform(GSR_feature_df)

'''
print("----------------愉悦度不使用交叉验证----------------")
train_X,test_X,train_Y,test_Y = \
    train_test_split(GSR_feature_df,all_df_y_valence,test_size=0.2,random_state=1000)
print("######linear regression######")
linearR_model = LinearRegression()
linearR_model.fit(train_X,train_Y)
linear_pred_Y = linearR_model.predict(test_X)
df_linear_pred_Y = pd.DataFrame(linear_pred_Y,columns=['valence'])
mse = mean_squared_error(linear_pred_Y,test_Y)
print("mse=",mse)
mae = mean_absolute_error(linear_pred_Y,test_Y)
print("mae=",mae)

print("######lasso model######")
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(train_X,train_Y)
lasso_pred_Y = lasso_model.predict(test_X)
df_lasso_pred_Y = pd.DataFrame(lasso_pred_Y,columns=['valence'])
mse = mean_squared_error(lasso_pred_Y,test_Y)
print("mse=",mse)
mae = mean_absolute_error(lasso_pred_Y,test_Y)
print("mae=",mae)

print("######ridge model######")
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(train_X,train_Y)
ridge_pred_Y = ridge_model.predict(test_X)
df_ridge_pred_Y = pd.DataFrame(ridge_pred_Y,columns=['valence'])
mse = mean_squared_error(ridge_pred_Y,test_Y)
print("mse=",mse)
mae = mean_absolute_error(ridge_pred_Y,test_Y)
print("mae=",mae)

print("######xgb(gbtree) regression model######")
dtrain = xgb.DMatrix(train_X,train_Y)      
dtest = xgb.DMatrix(test_X,test_Y)     
xgb_params = {
    'booster': 'gbtree',

    'eta': 0.1,

    'max_depth': 7,

    'objective': 'reg:linear',

    'eval_metric': 'mae',
    
    'colsample_bytree': 0.90,
    
    'alpha':0.6,
    
    'gamma':1,

    'silent':0,
}
watchlist = [(dtrain, 'train'), (dtest, 'test')]   
num_rounds = 50
#True 会使用watchlist
if False:   
    xgb_reg_model=xgb.train(xgb_params,dtrain,num_rounds,evals=watchlist)
else:
    xgb_reg_model=xgb.train(xgb_params,dtrain,num_rounds)
xgb_pred_Y = xgb_reg_model.predict(dtest)
mse = mean_squared_error(xgb_pred_Y,test_Y)
print("mse=",mse)
mae = mean_absolute_error(xgb_pred_Y,test_Y)
print("mae=",mae)

pickle.dump(xgb_pred_Y,open("./dump_file/xgb_pred_Y","wb"))
pickle.dump(test_Y,open("./dump_file/test_Y","wb"))
'''

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
lasso_model = Lasso(alpha=0.1)
lasso_scores = cross_val_score(lasso_model,data,target,cv=5,scoring='neg_mean_absolute_error')
print("lasso_scores:",abs(lasso_scores))
print("lasso_scores_mean:",abs(lasso_scores.mean()))

print("######ridge model CV######")
ridge_model = Ridge(alpha=0.1)
ridge_scores = cross_val_score(ridge_model,data,target,cv=5,scoring='neg_mean_absolute_error')
print("ridge_scores:",abs(ridge_scores))
print("ridge_scores_mean:",abs(ridge_scores.mean()))

print("######xgboost model CV######")
xgb_model  = xgb.XGBRegressor(max_depth=6,learning_rate=0.1,n_estimators=50,
                              objective='reg:linear',booster='gbtree',n_jobs=10,
                              subsample=1, colsample_bytree=0.9, colsample_bylevel=1,
                               reg_alpha=1.0, reg_lambda=1,gamma=1.0)
xgb_scores = cross_val_score(xgb_model,data,target,cv=5,scoring='neg_mean_absolute_error')
print("xgb_scores:",abs(xgb_scores))
print("xgb_scores_mean:",abs(xgb_scores.mean()))
#############################下面用唤醒度做回归#################################
print("----------------这是唤醒度预测----------------")
data = GSR_feature_df
target = all_df_y[['arousal']]
print("######linear regression CV######")
linearR_model = LinearRegression()
linearR_scores = cross_val_score(linearR_model,data,target,cv=5,scoring='neg_mean_absolute_error')
print("linearR_scores:",abs(linearR_scores))
print("linearR_scores_mean:",abs(linearR_scores.mean()))

print("######lasso model CV######")
lasso_model = Lasso(alpha=0.1)
lasso_scores = cross_val_score(lasso_model,data,target,cv=5,scoring='neg_mean_absolute_error')
print("lasso_scores:",abs(lasso_scores))
print("lasso_scores_mean:",abs(lasso_scores.mean()))

print("######ridge model CV######")
ridge_model = Ridge(alpha=0.1)
ridge_scores = cross_val_score(ridge_model,data,target,cv=5,scoring='neg_mean_absolute_error')
print("ridge_scores:",abs(ridge_scores))
print("ridge_scores_mean:",abs(ridge_scores.mean()))
'''
print("######SVR model CV######")
svr_model =  SVR()
svr_scores = cross_val_score(svr_model,data,target,cv=5,scoring='neg_mean_absolute_error')   
print("svr_scores:",svr_scores)
print("svr_scores_mean:",svr_scores.mean())
'''
print("######GridSearchCV######")
param_grid = {'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
lasso_model = Lasso()
gsearch = GridSearchCV(lasso_model,param_grid,cv=5)
gsearch.fit(data,target)

#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)
























