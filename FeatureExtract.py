"""
特征处理函数
"""
'''1-EEG特征'''
'''1.1-EEG时域特征'''
'''
1.1.1-EEG时域均值
IN：时域的离散数据值，比如8064个采样点
OUT：均值
'''
def eeg_mean(df):
    return df.mean(axis=1)
'''
1.1.2-EEG时域中值
IN：时域的离散数据值，比如8064个采样点
OUT：中值
'''
def eeg_median(df):
    return df.median(axis=1)
'''
1.1.2-EEG时域标准差
IN：时域的离散数据值，比如8064个采样点
OUT：标准差
'''
def eeg_std(df):
    return df.std(axis=1)
'''
1.1.2-EEG时域香农熵
IN：时域的离散数据值，比如8064个采样点
OUT：香农熵
'''