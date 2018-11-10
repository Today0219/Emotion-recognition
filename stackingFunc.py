"""
Created on Fri Nov  9 09:44:31 2018
stacking model fusion functions
@author: jinyx
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    skf = StratifiedKFold(n_splits=n_folds)
    
    i=0
    for (trainIdx, valiIdx) in skf.split(x_train,y_train):
        #print(x_train[trainIdx].shape,x_train[valiIdx].shape)        
        x_tra, y_tra = x_train[trainIdx], y_train[trainIdx]
        x_tst, y_tst =  x_train[valiIdx], y_train[valiIdx]
        clf.fit(x_tra, y_tra)
    
        second_level_train_set[valiIdx] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)
        i+=1
               
    #回归预测取均值，分类呢？
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

if __name__ == "__main__":
    X = [1,2,3,4,5,6,7,8,9,10]
    X = np.array(X)
    print(X)
    y = [0,0,0,0,1,1,1,1,1,1]
    get_stacking(clf=None,x_train=X ,y_train=y ,x_test=X,n_folds=2 )