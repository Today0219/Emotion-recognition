# =============================================================================
# .# -*- coding: utf-8 -*-
# =============================================================================
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from config import *

#32个实验者，每个实验者参与40个实验，每人共40路信号采集
sXX = ['s01','s02','s03','s04','s05','s06','s07','s08','s09',
       's10','s11','s12','s13','s14','s15','s16','s17','s18','s19',
       's20','s21','s22','s23','s24','s25','s26','s27','s28','s29',
       's30','s31','s32']

#read data from .dat files
for i in sXX:
    sXX_file_path ='./data_preprocessed_python/'+i+'.dat'
    f = open(sXX_file_path,'rb')
    locals()[i] = pickle.load(f, encoding='bytes')
    
#read labels 32 people(Y)
for i in sXX:
    locals()['%s_df_y'%i] = pd.DataFrame(locals()[i][b'labels'])
    locals()['%s_df_y'%i].columns = ['valence','arousal','dominance','liking'] 

#concat all sXX_df_y in one df
all_df_y = pd.DataFrame()
for i in sXX:
    temp_index = []
    for j in range(0,40,1):
        temp_index.append(i+'_'+str(j))
    locals()['%s_df_y'%i].index = temp_index
    all_df_y = pd.concat([all_df_y,locals()['%s_df_y'%i]],axis=0)

#index最终的表示方式例子：s01_0 ->(s01实验者 第0号情绪测量实验)
pickle.dump(all_df_y,open("./dump_file/all_df_y","wb"))

#############################提取32路EEG信号####################################
#read #32路EEG脑电信号,1到32路是脑电信号
for eeg_channel in range(1,33,1):
    for i in sXX:
        locals()['CH{}_{}_df_EEG_x'.format(eeg_channel,i)] = pd.DataFrame(locals()[i][b'data'][:][eeg_channel][:])
        temp_index = []
        for j in range(0,40,1):
            temp_index.append(i+'_'+str(j))
        locals()['CH{}_{}_df_EEG_x'.format(eeg_channel,i)].index = temp_index
    #concat all CHX_sXX_df_EEG_x in one df
    locals()['CH{}_df_EEG_x'.format(eeg_channel)] = pd.DataFrame()
    for i in sXX:
        locals()['CH{}_df_EEG_x'.format(eeg_channel)] = \
            pd.concat([locals()['CH{}_df_EEG_x'.format(eeg_channel)],locals()['CH{}_{}_df_EEG_x'.format(eeg_channel,i)]],axis=0)
    file_path = "./dump_file/{}".format('CH{}_df_EEG_x'.format(eeg_channel))
    pickle.dump(locals()['CH%s_df_EEG_x'%eeg_channel],open(file_path,"wb"))
###############################################################################
###########################提取1路GSR皮肤电信号################################
#read GSR data
for i in sXX:
    locals()['%s_df_GSR_x'%i] = pd.DataFrame(locals()[i][b'data'][:][36][:])
    temp_index = []
    for j in range(0,40,1):
        temp_index.append(i+'_'+str(j))
    locals()['%s_df_GSR_x'%i].index = temp_index

#concat all sXX_df_GSR_x in one df
all_df_GSR_x = pd.DataFrame()
for i in sXX:
    all_df_GSR_x = pd.concat([all_df_GSR_x,locals()['%s_df_GSR_x'%i]],axis=0)

pickle.dump(all_df_GSR_x,open("./dump_file/all_df_GSR_x","wb"))
###############################################################################

############################提取1路RSP呼吸信号#################################
#read Respiration belt data
for i in sXX:
    locals()['%s_df_RSP_x'%i] = pd.DataFrame(locals()[i][b'data'][:][37][:])
    temp_index = []
    for j in range(0,40,1):
        temp_index.append(i+'_'+str(j))
    locals()['%s_df_RSP_x'%i].index = temp_index

#concat all sXX_df_RSP_x in one df
all_df_RSP_x = pd.DataFrame()
for i in sXX:
    all_df_RSP_x = pd.concat([all_df_RSP_x,locals()['%s_df_RSP_x'%i]],axis=0)

pickle.dump(all_df_RSP_x,open("./dump_file/all_df_RSP_x","wb"))
###############################################################################

############################提取1路BVP信号#################################
#read Respiration belt data
for i in sXX:
    locals()['%s_df_BVP_x'%i] = pd.DataFrame(locals()[i][b'data'][:][38][:])
    temp_index = []
    for j in range(0,40,1):
        temp_index.append(i+'_'+str(j))
    locals()['%s_df_BVP_x'%i].index = temp_index

#concat all sXX_df_BVP_x in one df
all_df_BVP_x = pd.DataFrame()
for i in sXX:
    all_df_BVP_x = pd.concat([all_df_BVP_x,locals()['%s_df_BVP_x'%i]],axis=0)

pickle.dump(all_df_BVP_x,open("./dump_file/all_df_BVP_x","wb"))
###############################################################################

############################提取1路TMP信号#################################
#read Respiration belt data
for i in sXX:
    locals()['%s_df_TMP_x'%i] = pd.DataFrame(locals()[i][b'data'][:][39][:])
    temp_index = []
    for j in range(0,40,1):
        temp_index.append(i+'_'+str(j))
    locals()['%s_df_TMP_x'%i].index = temp_index

#concat all sXX_df_BVP_x in one df
all_df_TMP_x = pd.DataFrame()
for i in sXX:
    all_df_TMP_x = pd.concat([all_df_TMP_x,locals()['%s_df_TMP_x'%i]],axis=0)

pickle.dump(all_df_TMP_x,open("./dump_file/all_df_TMP_x","wb"))
###############################################################################

#################################画GSR信号的图##################################
#read .dat files(32 total)
f = open(s01_file_path,'rb')
s01 = pickle.load(f, encoding='bytes')
#s01_GSR_df_x,index:40 expriments,columns:8064 datas(128Hz)
s01_GSR_df_x = pd.DataFrame(s01[b'data'][:][36][:])
#s01_df_y,index:40 expriments,columns:Y
s01_df_y = pd.DataFrame(s01[ b'labels'])
s01_df_y.columns=['valence','arousal','dominance','liking']

plt.plot(s01_GSR_df_x.iloc[0,:])
plt.ylabel('GSR value')
plt.show()





































