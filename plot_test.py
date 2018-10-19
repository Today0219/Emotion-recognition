# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle 
import pandas as pd

all_df_y = pickle.load(open("./dump_file/all_df_y","rb"))
xgb_pred_Y = pickle.load(open("./dump_file/xgb_pred_Y","rb"))
test_Y = pickle.load(open("./dump_file/test_Y","rb"))
y_valence = pickle.load(open("./dump_file/y_valence","rb"))
y_arousal = pickle.load(open("./dump_file/y_arousal","rb"))
#画图DPI设定
DPI_SET = 100

if False:
    #########################愉悦度
    X = [1,2,3,4,5,6,7,8,9,10]
    Y_xgb = [1.675444,1.675363,1.680978,1.685009,1.691554,
             1.675277,1.674311,1.680342,1.691216,1.707188]
    Y_svr = [1.785470,1.785573,1.785693,1.785803,1.785853,
             1.785954,1.786118,1.786360,1.786597,1.786775]
    Y_ridge = [1.789497,1.790049,1.791245,1.792133,1.792958,1.793474,1.793847,
               1.794242,1.794706,1.795205]
    Y_linear = [1.76594288635,1.76594288635,1.76594288635,1.76594288635,1.76594288635,
                1.76594288635,1.76594288635,1.76594288635,1.76594288635,1.76594288635,]
    Y_xgb.sort()
    Y_svr.sort()
    Y_ridge.sort()
    Y_linear.sort()
    plt.figure(dpi = DPI_SET)
    plt.plot(X,Y_xgb,'ro',label="xgboost")
    plt.plot(X,Y_svr,'bs',label="SVR")
    plt.plot(X,Y_ridge,'g^',label="Ridge")
    plt.plot(X,Y_linear,'k+',label="OLS")
    plt.grid(False)
    plt.xlabel(u"最佳MAE表现模型排名")
    plt.ylabel("MAE")
    plt.title(u'愉悦度预测最佳十次模型对应的MAE')
    plt.legend()
    plt.show()
    #plt.savefig('MAE',dpi=100)
if False:
    #####################唤醒度
    X = [1,2,3,4,5,6,7,8,9,10]
    Y_xgb = [1.687146,1.695196,1.692517,1.699958,1.696151,
             1.689644,1.689434,1.689765,1.692192,1.690135]
    Y_svr = [1.709018,1.709057,1.709095,1.709111,1.709189,1.709237,1.709258,
             1.709278,1.709373,1.709496]
    Y_ridge = [1.697378,1.697711,1.698095,1.698543,1.699079,
               1.699739,1.700587,1.701745,1.703502,1.707373]
    Y_linear = [1.71203616905,1.71203616905,1.71203616905,1.71203616905,1.71203616905,
                1.71203616905,1.71203616905,1.71203616905,1.71203616905,1.71203616905,]
    Y_xgb.sort()
    Y_svr.sort()
    Y_ridge.sort()
    Y_linear.sort()
    plt.figure(dpi = DPI_SET)
    plt.plot(X,Y_xgb,'ro',label="xgboost")
    plt.plot(X,Y_svr,'bs',label="SVR")
    plt.plot(X,Y_ridge,'g^',label="Ridge")
    plt.plot(X,Y_linear,'k+',label="OLS")
    plt.grid(False)
    plt.xlabel(u"最佳MAE表现模型排名")
    plt.ylabel("MAE")
    plt.title(u'唤醒度预测最佳十次模型对应的MAE')
    plt.legend()
    plt.show()
    #plt.savefig('MAE',dpi=100)

if False:
    #corrs,愉悦度
    Y_corrs = [0.004346,0.005027,0.052310,0.027697,0.069725,0.043048,0.016981,
               0.011768,0.000304,0.033582,0.061108,0.056917,0.072331,0.065231,
               0.026672,0.024671,0.045470, 0.036574,0.062127,0.049776,0.024718,
               0.020680,0.082800,0.081184,0.076538,0.105749,0.108450,0.112452]
    X_features = [i for i in range(1,29,1)]
    plt.figure(dpi = DPI_SET)
    plt.bar(X_features,Y_corrs,color='red')
    plt.grid(False)
    plt.xlabel("特征编号")
    plt.ylabel("皮尔逊相关系数")
    plt.title("不同特征与愉悦度的皮尔逊相关系数")
    plt.legend()
    plt.show()

if False:
    #corrs,唤醒度
    Y_corrs = [0.048455,0.046454,0.023819,0.038387,0.018331,0.032164,0.025527,
               0.039727,0.042079,0.029262,0.015931,0.011253,0.002549,0.006947,
               0.003238,0.013122,0.022206,0.026620,0.014166,0.020778,0.001383,
               0.018818,0.014761,0.009047,0.023284,0.010995,0.000872,0.006976]
    X_features = [i for i in range(1,29,1)]
    plt.figure(dpi = DPI_SET)
    plt.bar(X_features,Y_corrs,color='red')
    plt.grid(False)
    plt.xlabel("特征编号")
    plt.ylabel("皮尔逊相关系数")
    plt.title("不同特征与唤醒度的皮尔逊相关系数")
    plt.legend()
    plt.show()

if False:
    #画出愉悦度，唤醒度的图
    #x = all_df_y[all_df_y['valence']>5][all_df_y['arousal']>5]['valence']
    #y = all_df_y[all_df_y['valence']>5][all_df_y['arousal']>5]['arousal']
    x_high = all_df_y[all_df_y['valence']>=5]['valence']
    y_high = all_df_y[all_df_y['valence']>=5]['arousal']
    x_low = all_df_y[all_df_y['valence']<=5]['valence']
    y_low = all_df_y[all_df_y['valence']<=5]['arousal']
    plt.figure(dpi = DPI_SET)
    plt.plot(x_high,y_high,'b.')
    plt.plot(x_low,y_low,'y.')
    plt.xlabel("愉悦度(valence)")
    plt.ylabel("唤醒度(arousal)")
    plt.title('样本愉悦度-唤醒度分布')
    plt.plot()

if True:
    #画出愉悦度，唤醒度的图,4个象限
    #x = all_df_y[all_df_y['valence']>5][all_df_y['arousal']>5]['valence']
    #y = all_df_y[all_df_y['valence']>5][all_df_y['arousal']>5]['arousal']
    x_1 = all_df_y[all_df_y['valence']>=5][all_df_y['arousal']>=5]['valence']
    y_1 = all_df_y[all_df_y['valence']>=5][all_df_y['arousal']>=5]['arousal']
    x_2 = all_df_y[all_df_y['valence']<5][all_df_y['arousal']>5]['valence']
    y_2 = all_df_y[all_df_y['valence']<5][all_df_y['arousal']>5]['arousal']
    x_3 = all_df_y[all_df_y['valence']<=5][all_df_y['arousal']<=5]['valence']
    y_3 = all_df_y[all_df_y['valence']<=5][all_df_y['arousal']<=5]['arousal']
    x_4 = all_df_y[all_df_y['valence']>5][all_df_y['arousal']<5]['valence']
    y_4 = all_df_y[all_df_y['valence']>5][all_df_y['arousal']<5]['arousal']
    plt.figure(dpi = DPI_SET)
    myMarkerSize = 3
    plt.plot(x_1,y_1,'b.',markersize=myMarkerSize)
    plt.plot(x_2,y_2,'y+',markersize=myMarkerSize)
    plt.plot(x_3,y_3,'gs',markersize=myMarkerSize)
    plt.plot(x_4,y_4,'r^',markersize=myMarkerSize)
    #plt.xlabel("愉悦度(valence)")
    #plt.ylabel("唤醒度(arousal)")
    plt.xlabel("valence")
    plt.ylabel("arousal")
    #plt.title('样本愉悦度-唤醒度分布')
    plt.show()

#统计样本个数
print("高愉悦度（5-9）个数：{}".format(len(x_high)))
print("高愉悦度（1-5）个数：{}".format(len(x_low)))

if False:
    #画3D图像
    x = all_df_y['valence']
    y = all_df_y['arousal']
    z = all_df_y['dominance'] 
    fig = plt.figure(dpi = DPI_SET)
    ax = Axes3D(fig)
    ax.scatter(x, y, z,'r.')
    ax.set_xlabel("愉悦度")
    ax.set_ylabel("唤醒度")
    ax.set_zlabel("支配度")
    ax.set_title("样本愉悦度-唤醒度-支配度分布")
    plt.show()

if False:
    #画箱线图，愉悦度
    df_test_Y = y_valence[['valence']]
    df_test_Y.columns=['valence_true']
    df_pred_Y = y_valence[['y_pred']]
    df_pred_Y.columns=['valence_pred']
    errors = abs(df_test_Y['valence_true'] - df_pred_Y['valence_pred'])
    df_errors = pd.DataFrame(errors,index=test_Y.index,columns=['abs_errors']) 
    df_result = pd.concat([df_test_Y,df_pred_Y],axis=1)
    df_result = pd.concat([df_result,df_errors],axis=1)
    for i in range(1,9):
        df_tmp = df_result[(df_result['valence_true']>=i) & (df_result['valence_true']<i+1)]
        locals()['errors_{}_{}'.format(i,i+1)] = abs(df_tmp['valence_true'] - df_tmp['valence_pred'])
        print(i)
    list_plt = [errors,errors_1_2,errors_2_3,errors_3_4,errors_4_5,errors_5_6,errors_6_7,errors_7_8,errors_8_9]
    list_labels =['总体误差','[1,2]','[2,3]','[3,4]','[4,5]','[5,6]','[6,7]','[7,8]','[8,9]']
    plt.figure(dpi = DPI_SET)
    plt.boxplot(list_plt,labels=list_labels)
    plt.ylabel("愉悦度误差MAE")
    plt.title("愉悦度回归预测误差箱线图")
    plt.show()

if False:
    #画箱线图，唤醒度
    df_test_Y = y_arousal[['arousal']]
    df_test_Y.columns=['arousal_true']
    df_pred_Y = y_arousal[['y_pred']]
    df_pred_Y.columns=['arousal_pred']
    errors = abs(df_test_Y['arousal_true'] - df_pred_Y['arousal_pred'])
    df_errors = pd.DataFrame(errors,index=test_Y.index,columns=['abs_errors']) 
    df_result = pd.concat([df_test_Y,df_pred_Y],axis=1)
    df_result = pd.concat([df_result,df_errors],axis=1)
    for i in range(1,9):
        df_tmp = df_result[(df_result['arousal_true']>=i) & (df_result['arousal_true']<i+1)]
        locals()['errors_{}_{}'.format(i,i+1)] = abs(df_tmp['arousal_true'] - df_tmp['arousal_pred'])
        print(i)
    list_plt = [errors,errors_1_2,errors_2_3,errors_3_4,errors_4_5,errors_5_6,errors_6_7,errors_7_8,errors_8_9]
    list_labels =['总体误差','[1,2]','[2,3]','[3,4]','[4,5]','[5,6]','[6,7]','[7,8]','[8,9]']
    plt.figure(dpi = DPI_SET)
    plt.boxplot(list_plt,labels=list_labels)
    plt.ylabel("唤醒度误差MAE")
    plt.title("唤醒度回归预测误差箱线图")
    plt.show()

if False:
    #画出不同区间内预测的精确度，愉悦度
    list_right = []
    list_total = []
    for i in range(1,9):
        num_right = y_valence['2C_pred_true'][(y_valence['valence'] >=i) & (y_valence['valence'] <i+1)].sum()
        num_total = len(y_valence['2C_pred_true'][(y_valence['valence'] >=i) & (y_valence['valence'] <i+1)])
        list_right.append(num_right)
        list_total.append(num_total)
        print("[%d],right:%d,total:%d,res:%f"%(i,num_right,num_total,num_right/num_total))
    list_x = ['总体误差','[1,2]','[2,3]','[3,4]','[4,5]','[5,6]','[6,7]','[7,8]','[8,9]']
    list_y = [sum(list_right)/sum(list_total)*100]
    for i in range(8):
        tmp_res = list_right[i]/list_total[i]*100
        list_y.append(tmp_res)
    plt.figure(dpi = DPI_SET)
    plt.bar(list_x,list_y,color='red')
    plt.grid(False)
    #plt.xlabel("特征编号")
    plt.ylabel("预测精度%")
    plt.title("愉悦度二分类预测精度")
    plt.legend()
    plt.show()

if False:
    #画出不同区间内预测的精确度，唤醒度
    list_right = []
    list_total = []
    for i in range(1,9):
        num_right = y_arousal['2C_pred_true'][(y_arousal['arousal'] >=i) & (y_arousal['arousal'] <i+1)].sum()
        num_total = len(y_arousal['2C_pred_true'][(y_arousal['arousal'] >=i) & (y_arousal['arousal'] <i+1)])
        list_right.append(num_right)
        list_total.append(num_total)
        print("[%d],right:%d,total:%d,res:%f"%(i,num_right,num_total,num_right/num_total))
    list_x = ['总体误差','[1,2]','[2,3]','[3,4]','[4,5]','[5,6]','[6,7]','[7,8]','[8,9]']
    list_y = [sum(list_right)/sum(list_total)*100]
    for i in range(8):
        tmp_res = list_right[i]/list_total[i]*100
        list_y.append(tmp_res)
    plt.figure(dpi = DPI_SET)
    plt.bar(list_x,list_y,color='red')
    plt.grid(False)
    #plt.xlabel("特征编号")
    plt.ylabel("预测精度%")
    plt.title("唤醒度二分类预测精度")
    plt.legend()
    plt.show()

if False:
    #画某个FFT之后的值
    scfft_df = pickle.load(open("./dump_file/scfft_df","rb"))
    plt.plot(scfft_df.iloc[0,:])






