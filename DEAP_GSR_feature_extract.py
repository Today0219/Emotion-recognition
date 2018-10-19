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

def sc_mean(df):
    return df.mean(axis=1)

def sc_median(df):
    return df.median(axis=1)

def sc_std(df):
    return df.std(axis=1)

def sc_min(df):
    return df.min(axis=1)

def sc_max(df):
    return df.max(axis=1)

def sc_range(df_max,df_min):
    return df_max['sc_max']-df_min['sc_min']

#最小值比率 = Mmin/N
def sc_minRatio(all_df,sc_min):
    all_df_T = all_df.T
    sc_min_T = sc_min.T
    sc_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len( all_df_T[i][ all_df_T[i] == sc_min_T.get_value(index='sc_min',col=i)] )   
        sc_minRatio_dict.update({i:num_min/8064.0})   
    sc_minRatio_df = pd.DataFrame.from_dict(data=sc_minRatio_dict,orient='index')
    sc_minRatio_df.columns = ['sc_minRatio']
    return sc_minRatio_df

#最大值比率 = Nmax/N
def sc_maxRatio(all_df,sc_max):
    all_df_T = all_df.T
    sc_max_T = sc_max.T
    sc_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len( all_df_T[i][ all_df_T[i] == sc_max_T.get_value(index='sc_max',col=i)] )   
        sc_maxRatio_dict.update({i:num_max/8064.0})    
    sc_maxRatio_df = pd.DataFrame.from_dict(data=sc_maxRatio_dict,orient='index')
    sc_maxRatio_df.columns = ['sc_maxRatio']
    return sc_maxRatio_df   

#GSR一阶差分均值 
def sc1Diff_mean(all_df):
    sc1Diff_mean = all_df.diff(periods=1,axis=1).dropna(axis=1).mean(axis=1)    
    return sc1Diff_mean

#GSR一阶差分中值
def sc1Diff_median(all_df):
    sc1Diff_median = all_df.diff(periods=1,axis=1).dropna(axis=1).median(axis=1)
    return sc1Diff_median

#GSR一阶差分标准差
def sc1Diff_std(all_df):
    sc1Diff_std = all_df.diff(periods=1,axis=1).dropna(axis=1).std(axis=1)
    return sc1Diff_std

def sc1Diff_min(all_df):
    sc1Diff_min = all_df.diff(periods=1,axis=1).dropna(axis=1).min(axis=1)
    return sc1Diff_min
    
def sc1Diff_max(all_df):
    sc1Diff_max = all_df.diff(periods=1,axis=1).dropna(axis=1).max(axis=1)
    return sc1Diff_max    

def sc1Diff_range(sc1Diff_max,sc1Diff_min):
    return sc1Diff_max['sc1Diff_max']-sc1Diff_min['sc1Diff_min']

def sc1Diff_minRatio(all_df,sc1Diff_min):
    all_df_Diff_T = all_df.diff(periods=1,axis=1).dropna(axis=1).T
    sc1Diff_min_T = sc1Diff_min.T
    sc1Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len( all_df_Diff_T[i][ all_df_Diff_T[i] == sc1Diff_min_T.get_value(index='sc1Diff_min',col=i)])
        sc1Diff_minRatio_dict.update({i:num_min/8063.0})
    sc1Diff_minRatio_df = pd.DataFrame.from_dict(data=sc1Diff_minRatio_dict,orient='index')
    return sc1Diff_minRatio_df

def sc1Diff_maxRatio(all_df,sc1Diff_max):
    all_df_Diff_T = all_df.diff(periods=1,axis=1).dropna(axis=1).T
    sc1Diff_max_T = sc1Diff_max.T
    sc1Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len( all_df_Diff_T[i][all_df_Diff_T[i] == sc1Diff_max_T.get_value(index='sc1Diff_max',col=i)])
        sc1Diff_maxRatio_dict.update({i:num_max/8063.0})
    sc1Diff_maxRatio_df = pd.DataFrame.from_dict(data=sc1Diff_maxRatio_dict,orient='index')
    return sc1Diff_maxRatio_df

def sc2Diff_std(all_df):
    sc2Diff_std = all_df.diff(periods=2,axis=1).dropna(axis=1).std(axis=1)
    return sc2Diff_std

def sc2Diff_min(all_df):
    sc2Diff_min = all_df.diff(periods=2,axis=1).dropna(axis=1).min(axis=1)
    return sc2Diff_min

def sc2Diff_max(all_df):
    sc2Diff_max = all_df.diff(periods=2,axis=1).dropna(axis=1).max(axis=1)
    return sc2Diff_max

def sc2Diff_range(sc2Diff_max,sc2Diff_min):
    sc2Diff_range = sc2Diff_max['sc2Diff_max']-sc2Diff_min['sc2Diff_min']
    return sc2Diff_range

def sc2Diff_minRatio(all_df,sc2Diff_min):
    all_df_2Diff_T = all_df.diff(periods=2,axis=1).dropna(axis=1).T
    sc2Diff_min_T = sc2Diff_min.T
    sc2Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len( all_df_2Diff_T[i][all_df_2Diff_T[i] == sc2Diff_min_T.get_value(index='sc2Diff_min',col=i)] )
        sc2Diff_minRatio_dict.update({i:num_min/8062.0})
    sc2Diff_minRatio_df = pd.DataFrame.from_dict(data=sc2Diff_minRatio_dict,orient='index')
    return sc2Diff_minRatio_df
        
def sc2Diff_maxRatio(all_df,sc2Diff_max):
    all_df_2Diff_T = all_df.diff(periods=2,axis=1).dropna(axis=1).T
    sc2Diff_max_T = sc2Diff_max.T
    sc2Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len( all_df_2Diff_T[i][all_df_2Diff_T[i] == sc2Diff_max_T.get_value(index='sc2Diff_max',col=i)] )
        sc2Diff_maxRatio_dict.update({i:num_max/8062.0})
    sc2Diff_maxRatio_df = pd.DataFrame.from_dict(data=sc2Diff_maxRatio_dict,orient='index')
    return sc2Diff_maxRatio_df

#GSR DFT(FFT)频域数据
def scfft(all_df):
    scfft_df = pd.DataFrame()
    for i in all_df_GSR_x.index.tolist():
        temp_scfft = pd.DataFrame(np.fft.fft(all_df_GSR_x.loc[i,:].values)).T
        temp_scfft.index = [i]
        scfft_df = scfft_df.append(temp_scfft)
    return scfft_df
        
#GSR 频域中值
def scfft_mean(scfft_df):
    scfft_mean = scfft_df.mean(axis=1)
    return scfft_mean

def scfft_median(scfft_df):
    scfft_median = scfft_df.median(axis=1)
    return scfft_median

def scfft_std(scfft_df):
    scfft_std = scfft_df.std(axis=1)
    return scfft_std

def scfft_min(scfft_df):
    scfft_min = scfft_df.min(axis=1)
    return scfft_min

def scfft_max(scfft_df):
    scfft_max = scfft_df.max(axis=1)
    return scfft_max

def scfft_range(scfft_max,scfft_min):
    scfft_range = scfft_max['scfft_max']-scfft_min['scfft_min']
    return scfft_range

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
    all_df_y = pickle.load(open("./dump_file/all_df_y","rb"))
    all_df_GSR_x = pickle.load(open("./dump_file/all_df_GSR_x","rb"))
    
    ###########################################################################
    if True :
        sc_mean = pd.DataFrame(sc_mean(all_df_GSR_x),columns=['sc_mean'])
        sc_median = pd.DataFrame(sc_median(all_df_GSR_x),columns=['sc_median'])
        sc_std = pd.DataFrame(sc_std(all_df_GSR_x),columns=['sc_std'])
        sc_min = pd.DataFrame(sc_min(all_df_GSR_x),columns=['sc_min'])
        sc_max = pd.DataFrame(sc_max(all_df_GSR_x),columns=['sc_max'])
        sc_range = pd.DataFrame(sc_range(sc_max,sc_min),columns=['sc_range'])
        sc_minRatio = pd.DataFrame(sc_minRatio(all_df_GSR_x,sc_min),columns=['sc_minRatio'])
        sc_maxRatio = pd.DataFrame(sc_maxRatio(all_df_GSR_x,sc_max),columns=['sc_maxRatio'])
        
        sc1Diff_mean = pd.DataFrame( sc1Diff_mean(all_df_GSR_x),columns=['sc1Diff_mean'])
        sc1Diff_median = pd.DataFrame( sc1Diff_median(all_df_GSR_x),columns=['sc1Diff_median'] )
        sc1Diff_std = pd.DataFrame( sc1Diff_std(all_df_GSR_x),columns=['sc1Diff_std'])
        sc1Diff_min = pd.DataFrame( sc1Diff_min(all_df_GSR_x),columns=['sc1Diff_min'])
        sc1Diff_max = pd.DataFrame( sc1Diff_max(all_df_GSR_x),columns=['sc1Diff_max'])
        sc1Diff_range = pd.DataFrame( sc1Diff_range(sc1Diff_max,sc1Diff_min),columns=['sc1Diff_range'])
        sc1Diff_minRatio = sc1Diff_minRatio(all_df_GSR_x,sc1Diff_min)
        sc1Diff_minRatio.columns=['sc1Diff_minRatio']
        sc1Diff_maxRatio = sc1Diff_maxRatio(all_df_GSR_x,sc1Diff_max)
        sc1Diff_maxRatio.columns=['sc1Diff_maxRatio']
        
        sc2Diff_std = pd.DataFrame( sc2Diff_std(all_df_GSR_x),columns=['sc2Diff_std'] )
        sc2Diff_min = pd.DataFrame( sc2Diff_min(all_df_GSR_x),columns=['sc2Diff_min'] ) 
        sc2Diff_max = pd.DataFrame( sc2Diff_max(all_df_GSR_x),columns=['sc2Diff_max'] )
        sc2Diff_range = pd.DataFrame(sc2Diff_range(sc2Diff_max,sc2Diff_min),columns=['sc2Diff_range'])
        sc2Diff_minRatio = sc2Diff_minRatio(all_df_GSR_x,sc2Diff_min)
        sc2Diff_minRatio.columns=['sc2Diff_minRatio']
        sc2Diff_maxRatio = sc2Diff_maxRatio(all_df_GSR_x,sc2Diff_max)
        sc2Diff_maxRatio.columns=['sc2Diff_maxRatio']

        if False:
            scfft_df = scfft(all_df_GSR_x)
            pickle.dump(scfft_df,open("./dump_file/scfft_df","wb"))
        else:
            scfft_df = pickle.load(open("./dump_file/scfft_df","rb"))
        
        scfft_mean = pd.DataFrame( scfft_mean(scfft_df),columns=['scfft_mean'])
        scfft_median = pd.DataFrame( scfft_median(scfft_df),columns=['scfft_median'])
        scfft_std = pd.DataFrame( scfft_std(scfft_df),columns=['scfft_std'])
        scfft_min = pd.DataFrame( scfft_min(scfft_df),columns=['scfft_min'])
        scfft_max = pd.DataFrame( scfft_max(scfft_df),columns=['scfft_max'])
        scfft_range = pd.DataFrame( scfft_range(scfft_max,scfft_min),columns=['scfft_range'])
       
        feature_list = ['sc_mean','sc_median','sc_std','sc_min','sc_max','sc_range',
                        'sc_minRatio','sc_maxRatio','sc1Diff_mean','sc1Diff_median',
                        'sc1Diff_std','sc1Diff_min','sc1Diff_max','sc1Diff_range',
                        'sc1Diff_minRatio','sc1Diff_maxRatio','sc2Diff_std',
                        'sc2Diff_min','sc2Diff_max','sc2Diff_range','sc2Diff_minRatio',
                        'sc2Diff_maxRatio','scfft_mean','scfft_median','scfft_std',
                        'scfft_min','scfft_max','scfft_range']
        temp_feature_df = pd.DataFrame()
        for i in feature_list:
            temp_feature_df = pd.concat( [locals()[i],temp_feature_df],axis=1)
            
        GSR_feature_df = temp_feature_df
        pickle.dump(GSR_feature_df,open("./dump_file/GSR_feature_df","wb"))
        ######################################################################

    if True:
        '''
        print(all_df_y)
        all_df_y_copy = all_df_y.copy()
        all_df_y_copy['emotion'] = 0
        all_df_y_copy['emotion'][ all_df_y_copy['valence'] >= 6] = 2
        all_df_y_copy['emotion'][ (all_df_y_copy['valence'] < 6) & (all_df_y_copy['valence'] >= 4)] = 1 
        all_df_y_copy['emotion'][ all_df_y_copy['valence'] < 4] = 0
        all_df_y_mutiLable = all_df_y_copy[['emotion']]
        pickle.dump(all_df_y_mutiLable,open("./dump_file/all_df_y_mutiLable","wb"))
        '''
        print(all_df_y)
        all_df_y_copy = all_df_y.copy()
        all_df_y_copy['emotion'] = 0
        all_df_y_copy['emotion'][ (all_df_y_copy['valence'] >= 5) & (all_df_y_copy['arousal'] >= 5)] = 0
        all_df_y_copy['emotion'][ (all_df_y_copy['valence'] < 5) & (all_df_y_copy['arousal'] >= 5)] = 1 
        all_df_y_copy['emotion'][ (all_df_y_copy['valence'] < 5) & (all_df_y_copy['arousal'] < 5)] = 2
        all_df_y_copy['emotion'][ (all_df_y_copy['valence'] >= 5) & (all_df_y_copy['arousal'] < 5)] = 3
        all_df_y_mutiLable = all_df_y_copy[['emotion']]
        pickle.dump(all_df_y_mutiLable,open("./dump_file/all_df_y_mutiLable","wb"))       
    
        all_df_y_copy = all_df_y.copy()
        all_df_y_copy['emotion_2'] = 0
        all_df_y_copy['emotion_2'][(all_df_y_copy['valence'] >= 5) & (all_df_y_copy['arousal'] >= 5)] = 1
        all_df_y_2c = all_df_y_copy[['emotion_2']]
        pickle.dump(all_df_y_2c,open("./dump_file/all_df_y_2c","wb"))   
    
        all_df_y_copy = all_df_y.copy()
        all_df_y_valence = all_df_y_copy[['valence']]
        pickle.dump(all_df_y_valence,open("./dump_file/all_df_y_valence","wb"))
    

        

    
    
    
    
    
    
    
    
    
    
    
    
 