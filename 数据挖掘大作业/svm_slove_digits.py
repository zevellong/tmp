# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:18:22 2021
SVM的运行速度和准确率测试
使用数据集手写数字识别
@author: longz
"""
import numpy as np
import sklearn.svm as svm
from sklearn import preprocessing
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multi_svm_using_numpy import demo_SVM, OVO_demo_SVM, OVR_demo_SVM

# 手写数字识别数据
def load_data():
    """
     加载SVM测试的二分类数据，将y转换成-1 1
    isShow : bool
        plot data scatter if true. The default is True.
    """
    #手写数字识别 1797x10, class_num = 10
    data = datasets.load_digits()
    # data = datasets.load_iris()
    all_X = data.data/16 #防止高斯核函数运算时指数爆炸
    # all_X= preprocessing.scale(all_X)
    all_y = data.target
    
    #8:2划分
    train_x, test_x, train_y, test_y = \
        train_test_split(all_X, all_y, test_size=0.2, 
                         random_state=0)
    
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = load_data()

def test_sk_svm():
    sk_svm = svm.SVC(random_state=0, kernel='rbf').fit(train_x, train_y)
    test_y_pred = sk_svm.predict(test_x)
    print("[sk]训练集Acc:{:.3%}".format((sk_svm.predict(train_x)==train_y).sum() / train_y.shape[0]))
    print("[sk]测试集Acc:{:.3%}".format((test_y_pred==test_y).sum() / test_y.shape[0]))

def test_OVO_svm():
    ovo_svm = OVR_demo_SVM(kname="gauss").fit(train_x, train_y)
    test_y_pred = ovo_svm.predict(test_x)
    print("[OVO_demo]训练集Acc:{:.3%}".format((ovo_svm.predict(train_x)==train_y).sum() / train_y.shape[0]))
    print("[OVO_demo]测试集Acc:{:.3%}".format((test_y_pred==test_y).sum() / test_y.shape[0]))


def main():
    test_sk_svm()
    test_OVO_svm()

    