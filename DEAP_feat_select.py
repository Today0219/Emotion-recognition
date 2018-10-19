import pandas as pd
import numpy as np
import pickle

def feat_select(use_GSR,use_RSP,use_EEG,complex_abs,complex_real,complex_imag):
    #读取原始特征
    GSR_feature_df = pickle.load(open("./dump_file/GSR_feature_df","rb"))
    RSP_feature_df = pickle.load(open("./dump_file/RSP_feature_df","rb"))    
    for eeg_CH in range(1,33,1):
        locals()["CH{}EEG_feature_df".format(eeg_CH)] = pickle.load(open("./dump_file/CH{}_eeg_feat_df".format(eeg_CH),"rb"))
    all_df_y_mutiLable = pickle.load(open("./dump_file/all_df_y_mutiLable","rb"))
    all_df_y_valence = pickle.load(open("./dump_file/all_df_y_valence","rb"))
    all_df_y= pickle.load(open("./dump_file/all_df_y","rb"))
    if use_GSR == False:
        GSR_feature_df = pd.DataFrame()
    if use_RSP == False:
        RSP_feature_df = pd.DataFrame()
    #把特征都合并在一起
    df_feat = pd.concat([GSR_feature_df,RSP_feature_df],axis=1)
    if use_EEG == True:
           for eeg_CH in range(1,33,1):
               df_feat = pd.concat([df_feat,locals()["CH{}EEG_feature_df".format(eeg_CH)]],axis=1)
    
    #复数的实数部分特征
    if complex_real == True:
        df_real = df_feat.select_dtypes(["complex128"]).apply(lambda x:x.real)
        list_new_col=[]
        for col in df_real.columns:
            list_new_col.append('real_{}'.format(col))
        df_real.columns = list_new_col
        df_feat = pd.concat([df_real,df_feat],axis=1)
        
    #复数的虚数部分特征
    if complex_imag == True:
        df_imag = df_feat.select_dtypes(["complex128"]).apply(lambda x:x.imag)
        list_new_col=[]
        for col in df_imag.columns:
            list_new_col.append('imag_{}'.format(col))
        df_imag.columns = list_new_col
        df_feat = pd.concat([df_imag,df_feat],axis=1)
    
    #True: drop complex data
    if complex_abs == False:
        if use_GSR == True:
            df_feat.drop(['scfft_mean','scfft_median','scfft_std',
                     'scfft_min','scfft_max','scfft_range'
                     ],inplace=True,axis=1)  
        if use_RSP == True:
            df_feat.drop(['rspfft_max',
                     'rspfft_range','rspfft_min','rspfft_median','rspfft_mean'
                     ],inplace=True,axis=1)
        if use_EEG == True:
            df_feat.drop(['CH2eeg2Diff_range','CH2eeg2Diff_max',
                         'CH2eeg2Diff_min','CH2eeg1Diff_range','CH2eeg1Diff_max','CH2eeg1Diff_min',
                         'CH2eeg1Diff_median','CH2eeg1Diff_mean','CH2eeg_range','CH2eeg_max',
                        'CH2eeg_min','CH2eeg_median','CH2eeg_mean'],inplace=True,axis=1)    
            for eeg_CH in range(1,33,1):
                df_feat.drop(['CH{}eegfft_mean'.format(eeg_CH),'CH{}eegfft_median'.format(eeg_CH),
                         'CH{}eegfft_std'.format(eeg_CH), 'CH{}eegfft_min'.format(eeg_CH),
                         'CH{}eegfft_max'.format(eeg_CH),'CH{}eegfft_range'.format(eeg_CH),],inplace=True,axis=1)
    
    elif complex_abs == True:
        #compute abs for complex        
        df_abs = df_feat.select_dtypes(["complex128"]).apply(np.abs)
        list_drop = df_abs.columns
        df_feat.drop(labels=list_drop,axis=1,inplace=True)
        df_feat = pd.concat([df_abs,df_feat],axis=1)
    

     
    df_feat_selected = df_feat
    #根据相关程度筛选数据
    if True:
        feature_cols = df_feat.columns
        #测试愉悦度
        corrs = df_feat[feature_cols].apply(lambda col:np.abs(all_df_y['valence'].corr(col)))
        #测试唤醒度
        #corrs = df_feat[feature_cols].apply(lambda col:np.abs(all_df_y['arousal'].corr(col)))
        sort_corrs = corrs.sort_values()
        selected_feature = sort_corrs[sort_corrs > 0.00].index
        df_feat_selected = df_feat[selected_feature]
        print(sort_corrs)      
    return df_feat_selected
        
if __name__ == '__main__':    
    df_feat_selected = feat_select(use_GSR=True,use_RSP=False,use_EEG=False,complex_abs=True,complex_real=False,complex_imag=False)
    #df_feat_selected = feat_select(use_GSR=True,use_RSP=False,use_EEG=False,complex_abs=True,complex_real=True,complex_imag=True)
    print('df_feat_selected.shape:',df_feat_selected.shape)

    pickle.dump(df_feat_selected,open("./dump_file/df_feat_selected","wb"))
    
    
    
    
    
    
    
    