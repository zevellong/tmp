# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:37:43 2021
绘图测试模块
测试核函数二分类，绘制边界和支持向量，对比线性核，高斯核，多项式核
测试多分类，1v1，1vR
@author: longz
"""

import numpy as np
import sklearn.svm as svm
from sklearn import preprocessing
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multi_svm_using_numpy import demo_SVM, OVO_demo_SVM, OVR_demo_SVM, load_data

h = 0.05
global figId
figId = 0
pre = "../figure/" #请修改图片保存路径前缀
figType = ".pdf" #保存图片的文件类型
def load_data_bin():
    iris = datasets.load_iris()
    all_X = iris.data[50:,[0,2]]
    all_y = iris.target[50:]
    all_y[all_y != 1] = -1
    train_x, test_x, train_y, test_y = \
        train_test_split(all_X, all_y, test_size=0.2, 
                         random_state=0)
    
    return train_x, test_x, train_y, test_y
    
def test_kernal_svm():
    train_x, test_x, train_y, test_y = load_data_bin()
    sk_test = ['linear', 'poly', 'rbf']
    de_test = ["linear", "mult", "gauss"]
    plt.figure()
    
    global figId
    for i in range(3):
        sk_svm =  svm.SVC(kernel=sk_test[i]).fit(train_x, train_y)
        de_svm =  demo_SVM(kname=de_test[i]).fit(train_x, train_y)
        xx, yy = np.meshgrid(np.arange(train_x[:,0].min(), train_x[:,0].max(), h),
                          np.arange(train_x[:,1].min(), train_x[:,1].max(), h))
        
        Z1 = sk_svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        Z2 = de_svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        Z1[Z1 == -1] = 2
        Z2[Z2 == -1] = 2
        plt.figure()
        plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
        plt.contourf(xx, yy, Z1, cmap=plt.cm.ocean, alpha=0.6)
        plt.title("sk_SVM, kernal="+sk_test[i])
        idx = sk_svm.support_
        plt.scatter(train_x[idx,0], train_x[idx,1], 
                    c='none',marker="o",edgecolors='black', s=100)
        plt.savefig(pre+str(figId)+"_kernal_svm_"+sk_test[i]+figType)
        figId += 1
        
        plt.figure(dpi=300)
        plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
        plt.contourf(xx, yy, Z2, cmap=plt.cm.ocean, alpha=0.6)
        plt.title("de_SVM, kernal="+ de_test[i])
        idx = de_svm.support_
        plt.scatter(train_x[idx,0], train_x[idx,1], 
                    c='none',marker="o",edgecolors='black', s=100)
        plt.savefig(pre+str(figId)+"_kernal_svm_"+de_test[i]+figType)
        figId += 1
        
def test_mult_svm():
    train_x, train_y,test_x,  test_y = load_data()
    lst = ['sk_SVM', 'demo_OVR', 'demo_OVO']
    es = [svm.SVC(kernel='rbf'), OVR_demo_SVM(kname='gauss'), OVO_demo_SVM(kname='gauss')]
    
    global figId
    for i in range(3):
        xx, yy = np.meshgrid(np.arange(train_x[:,0].min(), train_x[:,0].max(), h),
                          np.arange(train_x[:,1].min(), train_x[:,1].max(), h))
        es[i].fit(train_x, train_y)
        Z = es[i].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.figure()
        plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
        plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)
        plt.title(lst[i]+", kernal=Gauss")
        idx = es[i].support_
        plt.scatter(train_x[idx,0], train_x[idx,1], 
                    c='none',marker="o",edgecolors='black', s=100)
        plt.savefig(pre+str(figId)+"_"+lst[i]+figType)
        figId += 1
        
def main():
    test_kernal_svm()
    test_mult_svm()

if __name__ == '__main__':
    main()
        
    
    
