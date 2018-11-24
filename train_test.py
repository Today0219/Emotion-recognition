import pickle 
from sklearn.model_selection import train_test_split
#读取Y
all_df_y = pickle.load(open("./dump_file/all_df_y","rb"))
all_df_y['2cArousal'] = 0
all_df_y['2cArousal'][all_df_y['valence'] >= 5] = 1
all_df_y['2cValence'] = 0
all_df_y['2cValence'][all_df_y['valence'] >= 5] = 1
print(all_df_y.head(5))
#读取32个通道的EEG数据，每个通道包含32×40=1280个信号样本（人次×每人次40实验）
#每个样本向量大小为8064点（63s*128Hz）
for eegCH in range(1,2,1):
    file_path = "./dump_file/CH{}_df_EEG_x".format(eegCH)
    locals()['CH{}_df_EEG_x'.format(eegCH)] = pickle.load(open(file_path,"rb"))
    #file_path = "./dump_file/CH{}eegfft_df".format(eegCH)
    #locals()["CH{}eegfft_df".format(eegCH)] = pickle.load(open(file_path,"rb"))

X = CH1_df_EEG_x
y = all_df_y[['2cValence']]
if True:
    for seed in [0,100,200,300,400,500,600,700,800,900]:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,stratify=y,random_state=seed)   
        xTrainIdx = X_tr.index
        xTestIdx = X_te.index
        pickle.dump(xTrainIdx,open("./dump_file/xTrainIdx_{}".format(seed),"wb"))
        pickle.dump(xTestIdx,open("./dump_file/xTestIdx_{}".format(seed),"wb"))