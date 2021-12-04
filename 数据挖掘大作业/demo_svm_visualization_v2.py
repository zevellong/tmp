# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:45:12 2021

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
pre = "./figure/"
figType = ".pdf"

cmaps = ["terrain", "ocean"]
cmaps2 = ["twilight_shifted_r", "ocean", "ocean_r"]


def generate_planar_dataset():
    np.random.seed(709)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m)) # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.01 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    
    return X, Y

def load_data_bin():
    # iris = datasets.load_iris()
    all_X, all_y = generate_planar_dataset()
    all_y[all_y != 1] = -1
    train_x, test_x, train_y, test_y = \
        train_test_split(all_X, all_y, test_size=0.2, 
                         random_state=0)
    
    return train_x, test_x, train_y, test_y



def test_kernal_svm():
    train_x, test_x, train_y, test_y = load_data_bin()
    sk_test = ['linear', 'sigmoid', 'rbf']
    de_test = ["linear", "sigmoid", "gauss"]
    plt.figure()
    
    plt.scatter(train_x[:,0], train_x[:,1],c=train_y)
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
        plt.figure(dpi=300)
        
        plt.contourf(xx, yy, Z1, cmap="ocean", alpha=0.6)
        plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
        plt.title("sk_SVM, kernal="+sk_test[i])
        idx = sk_svm.support_
        plt.scatter(train_x[idx,0], train_x[idx,1], 
                    c='none',marker="o",edgecolors='black', s=100)
        plt.savefig(pre+str(figId)+ "svm" + figType)
        figId += 1
        
        plt.figure(dpi=300)
        
        plt.contourf(xx, yy, Z2,  cmap="terrain", alpha=0.6)
        plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
        plt.title("de_SVM, kernal="+ de_test[i])
        idx = de_svm.support_
        plt.scatter(train_x[idx,0], train_x[idx,1], 
                    c='none',marker="o",edgecolors='black', s=100)
        plt.savefig(pre+str(figId)+ "svm" +figType)
        figId += 1
        
def test_mult_svm(s=0):
    np.random.seed(s)
    train_x, train_y,test_x,  test_y = load_data()
    lst = ['sk_SVM', 'demo_OVR', 'demo_OVO']
    es = [svm.SVC(kernel='rbf'), OVR_demo_SVM(kname='gauss'), OVO_demo_SVM(kname='gauss')]
    
    global figId
    for i in range(3):
        xx, yy = np.meshgrid(np.arange(train_x[:,0].min(), train_x[:,0].max(), h),
                          np.arange(train_x[:,1].min(), train_x[:,1].max(), h))
        es[i].fit(train_x, train_y)
        Z = es[i].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.figure(dpi=300)
        plt.contourf(xx, yy, Z, cmap=cmaps2[i], alpha=0.6)
        plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
        plt.title(lst[i]+", kernal=Gauss")
        idx = es[i].support_
        plt.scatter(train_x[idx,0], train_x[idx,1], 
                    c='none',marker="o",edgecolors='black', s=100)
        plt.savefig(pre+str(figId)+ "svm"+ figType)
        figId += 1
        
def main():
    test_kernal_svm()
    test_mult_svm(s=31)

if __name__ == '__main__':
    main()
        
    
    
