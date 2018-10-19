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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
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
features_col = GSR_feature_df.columns
min_max_scaler = MinMaxScaler()
GSR_feature_ndarray = min_max_scaler.fit_transform(GSR_feature_df)
GSR_feature_df = pd.DataFrame(GSR_feature_ndarray)
GSR_feature_df.columns = features_col


#数据通过5折交叉验证划分
kf = KFold(n_splits=5)
k=[0,0,0,0,0]
k[0],k[1],k[2],k[3],k[4] = kf.split(GSR_feature_df)
##############################下面用交叉验证做愉悦度预测##################################
if True:
    print("##########愉悦度############")
    y_valence = all_df_y_valence.copy()
    df_predy = pd.DataFrame() #存放预测结果df
    MAE_sum = 0 #存放手动CV后的MAE和
    for i in range(0,5):
        data = GSR_feature_df.iloc[k[i][0]]
        target = all_df_y_valence.iloc[k[i][0]]
        test_x = GSR_feature_df.iloc[k[i][1]] 
        test_y = all_df_y_valence.iloc[k[i][1]]
        xgb_model  = xgb.XGBRegressor(max_depth=5,learning_rate=0.1,n_estimators=60,
                                      objective='reg:linear',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.1, reg_lambda=1.0,gamma=0)
        xgb_model.fit(data,target)
        test_predy = xgb_model.predict(test_x)
        MAE = mean_absolute_error(test_y,test_predy)
        MAE_sum = MAE + MAE_sum
        test_predy = pd.DataFrame(test_predy,columns=['y_pred'],index=test_y.index)
        df_predy = pd.concat([df_predy,test_predy],axis=0)
        print("[%d]MAE:%f"%(i,MAE))
    y_valence = pd.merge(y_valence,df_predy,how='outer',left_index=True,right_index=True)
    print("MAE_mean:%f"%(MAE_sum/5))
    y_valence['2C_pred_true']=0
    y_valence['2C_pred_true'][(y_valence['valence']>=5) & (y_valence['y_pred']>=5)] = 1  
    y_valence['2C_pred_true'][(y_valence['valence']<5) & (y_valence['y_pred']<5)] = 1
    accuracy = y_valence['2C_pred_true'].sum()/1280
    print("Accuracy:%f"%(accuracy))
    pickle.dump(y_valence,open("./dump_file/y_valence","wb"))

if False:
    print("##########唤醒度############")
    y_arousal = all_df_y[['arousal']].copy()
    df_predy = pd.DataFrame() #存放预测结果df
    MAE_sum = 0 #存放手动CV后的MAE和
    for i in range(0,5):
        data = GSR_feature_df.iloc[k[i][0]]
        target = all_df_y[['arousal']].iloc[k[i][0]]
        test_x = GSR_feature_df.iloc[k[i][1]] 
        test_y = all_df_y[['arousal']].iloc[k[i][1]]
        xgb_model  = xgb.XGBRegressor(max_depth=5,learning_rate=0.1,n_estimators=50,
                                      objective='reg:linear',booster='gbtree',n_jobs=10,
                                      subsample=0.90, colsample_bytree=0.90, colsample_bylevel=0.9,
                                      reg_alpha=0.1, reg_lambda=0.8,gamma=0)
        xgb_model.fit(data,target)
        test_predy = xgb_model.predict(test_x)
        MAE = mean_absolute_error(test_y,test_predy)
        MAE_sum = MAE + MAE_sum
        test_predy = pd.DataFrame(test_predy,columns=['y_pred'],index=test_y.index)
        df_predy = pd.concat([df_predy,test_predy],axis=0)
        print("[%d]MAE:%f"%(i,MAE))
    y_arousal = pd.merge(y_arousal,df_predy,how='outer',left_index=True,right_index=True)
    print("MAE_mean:%f"%(MAE_sum/5))
    y_arousal['2C_pred_true']=0
    y_arousal['2C_pred_true'][(y_arousal['arousal']>=5) & (y_arousal['y_pred']>=5)] = 1  
    y_arousal['2C_pred_true'][(y_arousal['arousal']<5) & (y_arousal['y_pred']<5)] = 1
    accuracy = y_arousal['2C_pred_true'].sum()/1280
    print("Accuracy:%f"%(accuracy))
    pickle.dump(y_arousal,open("./dump_file/y_arousal","wb"))

if False:
    y_arousal_2c = all_df_y[['arousal']].copy() 
    y_arousal_2c['2C'] = 0
    y_arousal_2c['2C'][y_arousal_2c['arousal'] >= 5] = 1
    df_predy = pd.DataFrame() #存放预测结果df
    for i in range(0,5):
        data = GSR_feature_df.iloc[k[i][0]]
        target = y_arousal_2c['2C'].iloc[k[i][0]]
        test_x = GSR_feature_df.iloc[k[i][1]] 
        test_y = y_arousal_2c['2C'].iloc[k[i][1]]
        xgb_model  = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,
                                      objective='binary:logistic',booster='gbtree',n_jobs=10,
                                      subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                                      reg_alpha=0.5, reg_lambda=1.0,gamma=0)
        xgb_model.fit(data,target)
        test_predy = xgb_model.predict(test_x)
        ACC = accuracy_score(test_y,test_predy)
        print("[%d]ACC:%f"%(i,ACC)) 
        test_predy = pd.DataFrame(test_predy,columns=['y_pred_2c'],index=test_y.index)
        df_predy = pd.concat([df_predy,test_predy],axis=0)
#用来计算程序运行时间
endtime = datetime.datetime.now()
print("程序运行时间:%.1fs"%(endtime - starttime).seconds)























