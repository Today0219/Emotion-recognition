# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:02:45 2018

@author: jinyx
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import pickle 
import matplotlib.pyplot as plt
from config import *
import warnings
warnings.filterwarnings("ignore")

def rsp_mean(df):
    return df.mean(axis=1)

def rsp_median(df):
    return df.median(axis=1)

def rsp_std(df):
    return df.std(axis=1)

def rsp_min(df):
    return df.min(axis=1)

def rsp_max(df):
    return df.max(axis=1)

def rsp_range(df_max,df_min):
    return df_max['rsp_max']-df_min['rsp_min']

#最小值比率 = Mmin/N
def rsp_minRatio(all_df,rsp_min):
    all_df_T = all_df.T
    rsp_min_T = rsp_min.T
    rsp_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len( all_df_T[i][ all_df_T[i] == rsp_min_T.get_value(index='rsp_min',col=i)] )   
        rsp_minRatio_dict.update({i:num_min/8064.0})   
    rsp_minRatio_df = pd.DataFrame.from_dict(data=rsp_minRatio_dict,orient='index')
    rsp_minRatio_df.columns = ['rsp_minRatio']
    return rsp_minRatio_df

#最大值比率 = Nmax/N
def rsp_maxRatio(all_df,rsp_max):
    all_df_T = all_df.T
    rsp_max_T = rsp_max.T
    rsp_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len( all_df_T[i][ all_df_T[i] == rsp_max_T.get_value(index='rsp_max',col=i)] )   
        rsp_maxRatio_dict.update({i:num_max/8064.0})    
    rsp_maxRatio_df = pd.DataFrame.from_dict(data=rsp_maxRatio_dict,orient='index')
    rsp_maxRatio_df.columns = ['rsp_maxRatio']
    return rsp_maxRatio_df   

#RSP一阶差分均值 
def rsp1Diff_mean(all_df):
    rsp1Diff_mean = all_df.diff(periods=1,axis=1).dropna(axis=1).mean(axis=1)    
    return rsp1Diff_mean

#RSP一阶差分中值
def rsp1Diff_median(all_df):
    rsp1Diff_median = all_df.diff(periods=1,axis=1).dropna(axis=1).median(axis=1)
    return rsp1Diff_median

#RSP一阶差分标准差
def rsp1Diff_std(all_df):
    rsp1Diff_std = all_df.diff(periods=1,axis=1).dropna(axis=1).std(axis=1)
    return rsp1Diff_std

def rsp1Diff_min(all_df):
    rsp1Diff_min = all_df.diff(periods=1,axis=1).dropna(axis=1).min(axis=1)
    return rsp1Diff_min
    
def rsp1Diff_max(all_df):
    rsp1Diff_max = all_df.diff(periods=1,axis=1).dropna(axis=1).max(axis=1)
    return rsp1Diff_max    

def rsp1Diff_range(rsp1Diff_max,rsp1Diff_min):
    return rsp1Diff_max['rsp1Diff_max']-rsp1Diff_min['rsp1Diff_min']

def rsp1Diff_minRatio(all_df,rsp1Diff_min):
    all_df_Diff_T = all_df.diff(periods=1,axis=1).dropna(axis=1).T
    rsp1Diff_min_T = rsp1Diff_min.T
    rsp1Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len( all_df_Diff_T[i][ all_df_Diff_T[i] == rsp1Diff_min_T.get_value(index='rsp1Diff_min',col=i)])
        rsp1Diff_minRatio_dict.update({i:num_min/8063.0})
    rsp1Diff_minRatio_df = pd.DataFrame.from_dict(data=rsp1Diff_minRatio_dict,orient='index')
    return rsp1Diff_minRatio_df

def rsp1Diff_maxRatio(all_df,rsp1Diff_max):
    all_df_Diff_T = all_df.diff(periods=1,axis=1).dropna(axis=1).T
    rsp1Diff_max_T = rsp1Diff_max.T
    rsp1Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len( all_df_Diff_T[i][all_df_Diff_T[i] == rsp1Diff_max_T.get_value(index='rsp1Diff_max',col=i)])
        rsp1Diff_maxRatio_dict.update({i:num_max/8063.0})
    rsp1Diff_maxRatio_df = pd.DataFrame.from_dict(data=rsp1Diff_maxRatio_dict,orient='index')
    return rsp1Diff_maxRatio_df

def rsp2Diff_std(all_df):
    rsp2Diff_std = all_df.diff(periods=2,axis=1).dropna(axis=1).std(axis=1)
    return rsp2Diff_std

def rsp2Diff_min(all_df):
    rsp2Diff_min = all_df.diff(periods=2,axis=1).dropna(axis=1).min(axis=1)
    return rsp2Diff_min

def rsp2Diff_max(all_df):
    rsp2Diff_max = all_df.diff(periods=2,axis=1).dropna(axis=1).max(axis=1)
    return rsp2Diff_max

def rsp2Diff_range(rsp2Diff_max,rsp2Diff_min):
    rsp2Diff_range = rsp2Diff_max['rsp2Diff_max']-rsp2Diff_min['rsp2Diff_min']
    return rsp2Diff_range

def rsp2Diff_minRatio(all_df,rsp2Diff_min):
    all_df_2Diff_T = all_df.diff(periods=2,axis=1).dropna(axis=1).T
    rsp2Diff_min_T = rsp2Diff_min.T
    rsp2Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len( all_df_2Diff_T[i][all_df_2Diff_T[i] == rsp2Diff_min_T.get_value(index='rsp2Diff_min',col=i)] )
        rsp2Diff_minRatio_dict.update({i:num_min/8062.0})
    rsp2Diff_minRatio_df = pd.DataFrame.from_dict(data=rsp2Diff_minRatio_dict,orient='index')
    return rsp2Diff_minRatio_df
        
def rsp2Diff_maxRatio(all_df,rsp2Diff_max):
    all_df_2Diff_T = all_df.diff(periods=2,axis=1).dropna(axis=1).T
    rsp2Diff_max_T = rsp2Diff_max.T
    rsp2Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len( all_df_2Diff_T[i][all_df_2Diff_T[i] == rsp2Diff_max_T.get_value(index='rsp2Diff_max',col=i)] )
        rsp2Diff_maxRatio_dict.update({i:num_max/8062.0})
    rsp2Diff_maxRatio_df = pd.DataFrame.from_dict(data=rsp2Diff_maxRatio_dict,orient='index')
    return rsp2Diff_maxRatio_df

#RSP DFT(FFT)频域数据
def rspfft(all_df):
    rspfft_df = pd.DataFrame()
    for i in all_df_RSP_x.index.tolist():
        temp_rspfft = pd.DataFrame(np.fft.fft(all_df_RSP_x.loc[i,:].values)).T
        temp_rspfft.index = [i]
        rspfft_df = rspfft_df.append(temp_rspfft)
    return rspfft_df
        
#RSP 频域中值
def rspfft_mean(rspfft_df):
    rspfft_mean = rspfft_df.mean(axis=1)
    return rspfft_mean

def rspfft_median(rspfft_df):
    rspfft_median = rspfft_df.median(axis=1)
    return rspfft_median

def rspfft_std(rspfft_df):
    rspfft_std = rspfft_df.std(axis=1)
    return rspfft_std

def rspfft_min(rspfft_df):
    rspfft_min = rspfft_df.min(axis=1)
    return rspfft_min

def rspfft_max(rspfft_df):
    rspfft_max = rspfft_df.max(axis=1)
    return rspfft_max

def rspfft_range(rspfft_max,rspfft_min):
    rspfft_range = rspfft_max['rspfft_max']-rspfft_min['rspfft_min']
    return rspfft_range

def get_123count(df):
    tmp_df =pd.DataFrame()
    for i in range(0,40,1):
        num_1 = len(df[i][ df[i]==1 ])
        num_2 = len(df[i][ df[i]==2 ])
        num_3 = len(df[i][ df[i]==3 ])
        list_num = [num_1,num_2,num_3]
        tmp_df = pd.concat([tmp_df,pd.DataFrame(list_num)],axis=1)  
    tmp_df.columns = range(0,40,1)
    tmp_df.index = ['num_1','num_2','num_3']
    return tmp_df
    
    
        
if __name__ == '__main__':
    #read file 
    all_df_RSP_x = pickle.load(open("./dump_file/all_df_RSP_x","rb"))
    
    ###########################################################################
    if True :
        rsp_mean = pd.DataFrame(rsp_mean(all_df_RSP_x),columns=['rsp_mean'])
        rsp_median = pd.DataFrame(rsp_median(all_df_RSP_x),columns=['rsp_median'])
        rsp_std = pd.DataFrame(rsp_std(all_df_RSP_x),columns=['rsp_std'])
        rsp_min = pd.DataFrame(rsp_min(all_df_RSP_x),columns=['rsp_min'])
        rsp_max = pd.DataFrame(rsp_max(all_df_RSP_x),columns=['rsp_max'])
        rsp_range = pd.DataFrame(rsp_range(rsp_max,rsp_min),columns=['rsp_range'])
        rsp_minRatio = pd.DataFrame(rsp_minRatio(all_df_RSP_x,rsp_min),columns=['rsp_minRatio'])
        rsp_maxRatio = pd.DataFrame(rsp_maxRatio(all_df_RSP_x,rsp_max),columns=['rsp_maxRatio'])
        
        rsp1Diff_mean = pd.DataFrame( rsp1Diff_mean(all_df_RSP_x),columns=['rsp1Diff_mean'])
        rsp1Diff_median = pd.DataFrame( rsp1Diff_median(all_df_RSP_x),columns=['rsp1Diff_median'] )
        rsp1Diff_std = pd.DataFrame( rsp1Diff_std(all_df_RSP_x),columns=['rsp1Diff_std'])
        rsp1Diff_min = pd.DataFrame( rsp1Diff_min(all_df_RSP_x),columns=['rsp1Diff_min'])
        rsp1Diff_max = pd.DataFrame( rsp1Diff_max(all_df_RSP_x),columns=['rsp1Diff_max'])
        rsp1Diff_range = pd.DataFrame( rsp1Diff_range(rsp1Diff_max,rsp1Diff_min),columns=['rsp1Diff_range'])
        rsp1Diff_minRatio = rsp1Diff_minRatio(all_df_RSP_x,rsp1Diff_min)
        rsp1Diff_minRatio.columns=['rsp1Diff_minRatio']
        rsp1Diff_maxRatio = rsp1Diff_maxRatio(all_df_RSP_x,rsp1Diff_max)
        rsp1Diff_maxRatio.columns=['rsp1Diff_maxRatio']
        
        rsp2Diff_std = pd.DataFrame( rsp2Diff_std(all_df_RSP_x),columns=['rsp2Diff_std'] )
        rsp2Diff_min = pd.DataFrame( rsp2Diff_min(all_df_RSP_x),columns=['rsp2Diff_min'] ) 
        rsp2Diff_max = pd.DataFrame( rsp2Diff_max(all_df_RSP_x),columns=['rsp2Diff_max'] )
        rsp2Diff_range = pd.DataFrame(rsp2Diff_range(rsp2Diff_max,rsp2Diff_min),columns=['rsp2Diff_range'])
        rsp2Diff_minRatio = rsp2Diff_minRatio(all_df_RSP_x,rsp2Diff_min)
        rsp2Diff_minRatio.columns=['rsp2Diff_minRatio']
        rsp2Diff_maxRatio = rsp2Diff_maxRatio(all_df_RSP_x,rsp2Diff_max)
        rsp2Diff_maxRatio.columns=['rsp2Diff_maxRatio']

        #FFT运算比较耗费时间，False直接读取跑过的文件
        if False:
            rspfft_df = rspfft(all_df_RSP_x)
            pickle.dump(rspfft_df,open("./dump_file/rspfft_df","wb"))
        else:
            rspfft_df = pickle.load(open("./dump_file/rspfft_df","rb"))
        
        rspfft_mean = pd.DataFrame( rspfft_mean(rspfft_df),columns=['rspfft_mean'])
        rspfft_median = pd.DataFrame( rspfft_median(rspfft_df),columns=['rspfft_median'])
        rspfft_std = pd.DataFrame( rspfft_std(rspfft_df),columns=['rspfft_std'])
        rspfft_min = pd.DataFrame( rspfft_min(rspfft_df),columns=['rspfft_min'])
        rspfft_max = pd.DataFrame( rspfft_max(rspfft_df),columns=['rspfft_max'])
        rspfft_range = pd.DataFrame( rspfft_range(rspfft_max,rspfft_min),columns=['rspfft_range'])
       
        feature_list = ['rsp_mean','rsp_median','rsp_std','rsp_min','rsp_max','rsp_range',
                        'rsp_minRatio','rsp_maxRatio','rsp1Diff_mean','rsp1Diff_median',
                        'rsp1Diff_std','rsp1Diff_min','rsp1Diff_max','rsp1Diff_range',
                        'rsp1Diff_minRatio','rsp1Diff_maxRatio','rsp2Diff_std',
                        'rsp2Diff_min','rsp2Diff_max','rsp2Diff_range','rsp2Diff_minRatio',
                        'rsp2Diff_maxRatio','rspfft_mean','rspfft_median','rspfft_std',
                        'rspfft_min','rspfft_max','rspfft_range']
        temp_feature_df = pd.DataFrame()
        for i in feature_list:
            temp_feature_df = pd.concat( [locals()[i],temp_feature_df],axis=1)
            
        RSP_feature_df = temp_feature_df
        pickle.dump(RSP_feature_df,open("./dump_file/RSP_feature_df","wb"))
        ######################################################################


        

    
    
    
    
    
    
    
    
    
    
    
    
 