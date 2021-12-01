# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 11:11:53 2021
SVM的numpy实现，接口模仿sklearn

_kernal 向量化的核函数
    - __call__(X, Z) X为预测的输入，Z为支持向量机中的X，X Z的第二个维度为特征，要求特征数相同
    - set_func 设置核函数，支持 字符串["linear", 
                                    "mult", "gauss", 
                                    "laplace", "sigmoid", "rbf"]
            支持参数重载：sigma，gamma，p为幂，c为偏置
demo_svm
    - fit(X, y) X矩阵，y先转换1和-1
    - predict(X) 输入X
    
OVR_demo_SVM 一对其他，先替换标签并记录，预测后替换回标签， n个类需要n个分类器，一维列表存贮
    - fit(X, y) X矩阵，y标签
    - predict(X) 输入X
    
OVO_demo_SVM 一对一，先替换标签，n个类需要 (n^2 - n)/2，使用下三角存贮分类器
    - fit(X, y) X矩阵，y标签
    - predict(X) 输入X，预测标签
    - loc(i,j) 找到对应的下三角位置的分类器，注意：i<j 不会做检查
@author: longz
"""

import numpy as np
import sklearn.svm as svm
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(isShow=False):
    """
     加载SVM测试的二分类数据，将y转换成-1 1
    isShow : bool
        plot data scatter if true. The default is True.
    """
    #iris 数据集 150x4，3个类别
    iris = datasets.load_iris()
    all_X = iris.data[:,[0,2]]
    all_y = iris.target[:]
    
    #8:2划分
    train_x, test_x, train_y, test_y = \
        train_test_split(all_X, all_y, test_size=0.2, 
                         random_state=0)
    if (isShow):
        plt.scatter(all_X[:,0], all_X[:,1], c=all_y)    
        plt.title("iris scatter")
        plt.show()
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = load_data()

def test_sk_svm(): 
    h = 0.05
    sk_svm = svm.SVC(random_state=0, kernel='rbf').fit(train_x, train_y)
    test_y_pred = sk_svm.predict(test_x)
    print("test_sk_svm:{:.3%}".format((test_y_pred==test_y).sum() / test_y.shape[0]))
    xx, yy = np.meshgrid(np.arange(train_x[:,0].min(), train_x[:,0].max(), h),
                     np.arange(train_x[:,1].min(), train_x[:,1].max(), h))
    plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
    Z = sk_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)
    plt.title("sk_svm")
    plt.show()
    return sk_svm

from numpy.linalg import norm 
# class _kernel(): #非向量化核函数
#     def __init__(self, func="linear"):
#         self.funcMode = ["linear", "mult", "gauss", "laplace", "sigmoid"]
#         self.sigma, self.p, self.gamma, self.c = 1, 2, 1, 0
#         # x @ y 表示矩阵乘法，如果xy都是一维向量，则计算内积
#         self.funcDict = {"linear": lambda x,y: x @ y,
#                          "mult": lambda x,y: (x @ y+self.c)**self.p,
#                          "gauss": lambda x,y: np.exp(- ((x-y) @ (x-y)) / self.sigma),
#                          "laplace": lambda x,y: np.exp(- (norm(x-y)) / self.sigma),
#                          "sigmoid": lambda x,y: np.tanh(self.gamma * x @ y + self.c)}
#         self.func = None
#         self.setFunc(func, sigma=1)

#     #设置核函数核参数
#     def setFunc(self, func, sigma=1, p=2, gamma=1/2, c=0):
#         if (func not in self.funcMode): #不支持的核函数将会终止
#             raise TypeError("Kernel function %s not supported"%(func))
#         self.sigma = sigma
#         self.p = p
#         self.gamma = gamma
#         self.c = c
#         self.func = self.funcDict[func]
        
#     # 重载函数调用运算符，先检查xy是否符合一维向量，然后调用核函数
#     def __call__(self, x, y):
#         if (False): # 未检查，后面再加
#             raise TypeError("In kernel calc:input shape is error")
#         return self.func(x, y)
class _kernel(): #向量化核函数
    def __init__(self, func="linear"):
        self.funcMode = ["linear", "mult", "gauss", "laplace", "sigmoid", "rbf"]
        self.sigma, self.p, self.gamma, self.c = 1, 2, 1, 0
        # x @ y 表示矩阵乘法，如果xy都是一维向量，则计算内积
        self.funcDict = {"linear": lambda X,Z: X @ Z.T,
                         "mult": lambda X,Z: (X @ Z.T +self.c)**self.p,
                         "gauss": lambda X,Z: self.__gauss(X, Z, self.sigma),
                         "laplace": lambda X,Z: self.__laplace(X, Z, self.sigma),
                         "rbf": lambda X,Z: self.__gauss(X, Z, 1/self.sigma),
                         "sigmoid": lambda X,Z: np.tanh(self.gamma * X @ Z.T + self.c)}
        self.func = None
        self.setFunc(func, sigma=2)       
    @staticmethod
    def __laplace(X, Z, sigma):
        sz = X.shape[0]
        rtn = np.zeros((sz, Z.shape[0]))
        for i in range(sz):
            rtn[i] = np.exp(- (norm(Z - X[i], axis=-1) / sigma))
        return rtn
        
    @staticmethod
    def __gauss(X, Z, sigma):
        sz = X.shape[0]
        rtn = np.zeros((sz, Z.shape[0]))
        for i in range(sz):
            rtn[i] = np.exp(-(((Z - X[i])**2).sum(axis=-1) / sigma))
        # print(-(((Z - X[i])**2).sum(axis=-1)).min(), -(((Z - X[i])**2).sum(axis=-1)).max())
        return rtn
        
    #设置核函数核参数
    def setFunc(self, func, sigma=1, p=2, gamma=1, c=0):
        if (func not in self.funcMode): #不支持的核函数将会终止
            raise TypeError("Kernel function %s not supported"%(func))
        self.sigma = sigma
        self.p = p
        self.gamma = gamma
        self.c = c
        self.func = self.funcDict[func]
        
    # 重载函数调用运算符 X为预测输入，Z为支持向量机存贮的X
    def __call__(self, X, Z):
        # if (X.shape[1] != Z.shape[1]): # 输入矩阵检查，不加可以略微提升速度（5s提升0.1s左右）
        #     raise TypeError("In kernel calc:input shape is error")
        return self.func(X, Z)
    
#demo SVM， 接口模仿sklearn
class demo_SVM():
    def __init__(self, kname="linear", C=1):
        self.kn = _kernel(func=kname)# 核函数
        self.knLinear = True if kname == 'linear' else False #是否使用线性核函数，便于使用W加速
        self.kcache = np.zeros((0)) #核函数计算的缓存
        self.C = C #惩罚系数
        self.W, self.b = None, 0 #超平面参数，仅适用线性核函数
        self.ecache = None #预测值与真实值之差
        self.support_ = None #支持向量集合
        self.alphas = None 
        self.eps = 1e-6 #误差容忍
        self.jcache = None 
        self.y = None #保存标签，predict将会使用到
        self.X_sup = None #保存支持向量的X，predict将会使用到
        self.m = 0 #输入的规模
        
    def fit(self, X, y):
        # kn = self.kn
        m = X.shape[0]
        # 先计算核函数缓存， 非向量化核函数计算缓存
        # Cache = np.zeros((m,m))
        # for i in range(m):
        #     for j in range(i,m):
        #         Cache[i,j] = Cache[j,i] = kn(X[i], X[j])
        self.kcache = self.kn(X, X) #计算核函数缓存
        self.m = m
        
        #初始化alpha        
        self.alphas = np.zeros((m,))
        #初始化E
        self.updateE(y)
        
        self.jcache = np.zeros((m,))
        self.y = y

        self.smo(X, y) #smo主流程
        
        #缓存W方便线性核加速
        self.W = ((self.alphas * y)[:,None].T @ X)[0]
        #得到支持向量
        self.support_ = np.nonzero(((self.alphas > self.eps) * (self.alphas < self.C + self.eps)))[0]
        self.X_sup = X[self.support_]
        self.clearCache()
        return self
    def clearCache(self):
        self.ecache = None 
        self.kcache = None
        self.jcache = None
    def predict(self, X, _hard = None):
        rtn = self.predictHard(X) if _hard is None else _hard
        rtn = (rtn >= 0).astype('int')
        rtn[rtn == 0] = -1
        return rtn
    
    def updateE(self, y):
        # 计算所有维度的E
        # m = y.shape[0]
        # if self.ecache is None:
        #     self.ecache = np.zeros((m,))
        # tmp = (self.alphas * y)
        # ecache = np.zeros((m,))
        # Cache = self.kcache
        # for i in range(m):
        #     ecache[i] = tmp @ Cache[:,i] + self.b - y[i]
        # self.ecache = ecache
        
        #向量化更新E，由于非支持向量的项不会影响计算结果（alpha=0，相加后不会影响结果），先拿到支持向量可以加速运算
        idx = (self.alphas != 0)
        self.ecache = self.b - y
        if (idx.any()):
            # print((self.kcache[:,idx] @ (self.alphas[idx] * y[idx])).shape)
            self.ecache += (self.kcache[:,idx] @ (self.alphas[idx] * y[idx]))

        
    @staticmethod #裁剪
    def _clip(a_new, L, H):
        if (a_new < L):
            return L
        elif (a_new > H):
            return H
        else:
            return a_new
    
    def innerL(self, i, y):
        C, eps = self.C, self.eps
        ai, Ei, yi = self.alphas[i], self.ecache[i], y[i]
       
        #满足KKT条件则返回，不需要优化
        if (eps < ai < C - eps and np.abs(yi * Ei) < eps) or \
           (ai < eps and yi * Ei >= 0) or \
           (ai > C - eps and  yi * Ei <= 0):
               return 0
        else: #否则进行优化返回，如果修改了一对ij则返回1
            j = self.selectJ(i)
            aj, Ej, yj = self.alphas[j], self.ecache[j], y[j]
            k_ii, k_jj, k_ij = self.kcache[i,i], self.kcache[j,j], self.kcache[i,j]
            eta = k_ii + k_jj - 2*k_ij
            if (eta <= 0):
                print("eta<=0")
                return 0
            
            #计算新alphaj
            aj_new = aj + yj*(Ei - Ej) / eta
            if (yi != yj):
                L = max(0, aj - ai)
                H = min(C, C + aj - ai)
            else:
                L = max(0, aj + ai- C)
                H = min(C, aj + ai)
             
            #裁剪
            aj_new = self._clip(aj_new, L, H)
            ai_new = max(0, ai + yi*yj*(aj - aj_new))
                
            if np.abs(aj_new - aj) < eps:
                return 0
            #更新alphai
            self.alphas[i], self.alphas[j] = ai_new, aj_new
            # 更新b
            bi = -Ei - yi*k_ii*(ai_new - ai) - yj*k_ij*(aj_new - aj) + self.b
            bj = -Ej - yi*k_ij*(ai_new - ai) - yj*k_jj*(aj_new - aj) + self.b
            if  self.eps < ai_new < C:
                b = bi
            elif self.eps < aj_new < C:
                b = bj
            else:
                b = (bi + bj) / 2
            self.b = b
            #重新计算E
            self.updateE(y)
            self.jcache[[i,j]] = 1
            
            return 1
    def selectJ(self, i):
        idx = np.nonzero(self.jcache)[0] #可能寻找j的集合
        # idx = np.hstack((idx, np.nonzero(self.alphas)[0]))
        # idx = np.arange(80)
        self.jcache[i] = 1
        if (idx.size != 0):
            Ei = self.ecache[i]
            if (Ei > 0): #快速找到影响最大的j，Ei小于0返回最大值；反之返回最小值
                return idx[np.argmin(self.ecache[idx])]
            else:
                return idx[np.argmax(self.ecache[idx])]
            
            #遍历式寻找，速度较慢
            # maxj, maxDiff = -1, 0
            # for j in idx:
            #     if (maxDiff < np.abs(Ei - self.ecache[j])):
            #         maxj, maxDiff = j, np.abs(Ei - self.ecache[j])
            # return maxj
        else: 
            return self.selectJRand(i)

    def selectJRand(self, i):
        j = np.random.choice(self.m)
        if (j == i):
            j = np.random.choice(self.m)
        return j
    
    def smo(self, X, y, itMax = 1000):
        it = 0
        #遍历边界还是非边界，有无alpha对改变
        searchBound, alphaPairsChanged = False, 0
        m,n = X.shape
        C = self.C

        #最大迭代次数 搜索全集没有改动 则退出
        while (it < itMax) and  ((alphaPairsChanged > 0) or (not searchBound)): 
            alphaPairsChanged = 0
            if (searchBound): #边界搜索
                Bounds =  np.nonzero((self.alphas > 0) * (self.alphas < C))[0]
                np.random.shuffle(Bounds)
                for i in Bounds:
                    alphaPairsChanged += self.innerL(i, y)
            else: #全集搜索
                shufRG =  np.arange(m)
                np.random.shuffle(shufRG)
                for i in shufRG: 
                    alphaPairsChanged += self.innerL(i, y)
            it += 1

            if (not searchBound): #上一次是全集搜索
                searchBound = True #改为边界搜索
            elif (alphaPairsChanged == 0): #aplha对没有改变
                searchBound = False #在全集中搜索
    def getWb(self):
        return self.W, self.b
    def predictHard(self, X): 
        if (self.knLinear):
            rtn = (X @ self.W + self.b)
        else:
            idx = self.support_
            kccc = self.kn(X,self.X_sup) #向量化的核函数转换输入
            
            # 非向量化的核函数将转换输入
            # print(kccc.shape, X.shape)
            # kccc = np.zeros((X.shape[0], idx.shape[0]))
            # for i in range(X.shape[0]):
            #     for j in range(len(idx)):
            #         kccc[i,j] =self.kn(X[i], self.X[idx[j]].T)
            rtn = (self.alphas[idx] * self.y[idx]) @ kccc.T + self.b #
            # print(self.alphas[idx] * train_y[idx])
        return rtn


def test_kernel_svm():
    iris = datasets.load_iris()
    d_svm = demo_SVM(kname='gauss')
    tx = iris.data[50:, [0,2]]
    ty = iris.target[50:]
    ty[ty == 2] = -1
    d_svm.fit(tx, ty)
    
    x = d_svm.predict(tx)
    # x[x == -1] = 2
    print("kernal_svm:{:.3%}".format((x == ty).sum() / len(x)))
    h = 0.05
    xx, yy = np.meshgrid(np.arange(tx[:,0].min(), tx[:,0].max(), h),
                     np.arange(tx[:,1].min(), tx[:,1].max(), h))
    plt.scatter(tx[:,0], tx[:,1], c=ty)
    Z = d_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z[Z == -1] = 2
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)
    idx = list(d_svm.support_)
    plt.scatter(tx[idx,0], tx[idx,1], c='none',marker="o",edgecolors='g', s=100)
    plt.title("m_svm")
    plt.show()
    
# test_kernel_svm()
                
class OVR_demo_SVM(): #将二分类SVM扩展到多分类,1对其他
    def __init__(self, kname = 'linear'):
        
        self.n_cls = None #类的数目
        self.idy_cls = None #将id转换为类标签
        self.es_lst = None #存贮分类器
        self.kn = kname #使用的核函数
        self.support_ = None #记录分类器的支持向量
    def predict(self, X):
        cls_score = np.zeros((X.shape[0], len(self.es_lst)))
        cls_tab = np.zeros((X.shape[0], len(self.es_lst)))
        for i in range(self.n_cls):
            predScore = self.es_lst[i].predictHard(X)
            pred = self.es_lst[i].predict(X, predScore)
            cls_score[:, i] = predScore
            cls_tab[pred==1, i] += 1
        
        y = np.zeros((X.shape[0],))
        for i in range(X.shape[0]):
            tmp = cls_tab[i]
            maxIdx = np.argmax(tmp)
            maxCt = (tmp == tmp[maxIdx]).sum()
            if (maxCt == 1): #所有的分类器预测的结果唯一
                y[i] = maxIdx
            else: #分类器预测结果不唯一，根据偏离的程度再预测
                maxIdx = np.argwhere(tmp == tmp[maxIdx])
                maxIdxIdx = np.argmax(np.abs(cls_score[i, maxIdx]))
                # print(maxIdx[maxIdxIdx], maxIdx.T)
                y[i] = maxIdx[maxIdxIdx]
        #将y替换为输入时候的标签
        rtny = np.zeros(y.shape)
        for i in range(self.n_cls):
            rtny[y == i] = self.idy_cls[i]
        return rtny
    def fit(self, X, y): #一对多多分类
        uniqY = np.unique(y)    
        self.n_cls = len(uniqY)
        # self.cls_idx = {k:v for k,v in zip(uniqY, range(self.n_cls))}
        self.idy_cls = uniqY #得到标签
        idy = np.zeros((y.shape))
        
        #将标签替换成0 1 2 3 ... 
        for i in range(self.n_cls):
            idy[y == uniqY[i]] = i
        
        self.support_ = np.zeros(())
        self.es_lst = []
        for i in range(self.n_cls):
            biny = -np.ones(idy.shape)
            biny[idy==i] = 1 
            # print("fit", i)
            self.es_lst.append(demo_SVM(kname=self.kn).fit(X, biny)) #调用fit
            self.support_ = np.hstack((self.support_, self.es_lst[i].support_)) #记录支持向量
        self.support_ = np.unique(self.support_).astype('int') 
        return self

def test_OVR_svm():
    h = 0.05
    m_svm = OVR_demo_SVM(kname='gauss')
    m_svm.fit(train_x, train_y)
    test_y_pred = m_svm.predict(test_x)
    print("OVR_svm:{:.3%}".format((test_y_pred==test_y).sum() / test_y.shape[0]))
    xx, yy = np.meshgrid(np.arange(train_x[:,0].min(), train_x[:,0].max(), h),
                      np.arange(train_x[:,1].min(), train_x[:,1].max(), h))
    plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
    Z = m_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z[Z == -1] = 2
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)
    idx = m_svm.support_
    plt.scatter(train_x[idx,0], train_x[idx,1], 
                c='none',marker="o",edgecolors='black', s=100)
    plt.title("OVR_svm")
    plt.show()


class OVO_demo_SVM(): #将二分类SVM扩展到多分类, 1v1多分类
    def __init__(self, kname = 'linear'):
        self.n_cls = None
        self.idy_cls = None
        self.cls_idy = None
        self.cls_tab = None
        self.es_lst = None
        self.kn = kname
        self.support_ = None
    def predict(self, X):
        cls_tab = np.zeros((X.shape[0], self.n_cls))
        for i in range(self.n_cls): #分类器投票制预测
            for j in range(i+1, self.n_cls):
                pred = self.es_lst[self.loc(i, j)].predict(X)
                cls_tab[pred==1, i] += 1
                cls_tab[pred!=1, j] += 1
        y = np.argmax(cls_tab, axis=1)
        rtny = y.copy()
        #id替换回原来的标签
        for i in range(self.n_cls):
            rtny[y == i] = self.idy_cls[i]
        return rtny
    
    def loc(self, i, j): #输入两个类id，找到对应分类器的下标, 上三角矩阵一维存储
        return (i * (2*self.n_cls - i + 1) // 2) + j - 2*i - 1
    
    def fit(self, X, y): #一对多多分类
        uniqY = np.unique(y)    
        self.n_cls = len(uniqY)
        self.idy_cls = uniqY #id找到类标签
        idy = np.zeros((y.shape))
        for i in range(self.n_cls):
            idy[y == uniqY[i]] = i
        self.es_lst = []
        self.support_ = np.zeros(())
        for i in range(self.n_cls):
            for j in range(i+1, self.n_cls):
                # print(i,j)
                idx = np.logical_or(idy==i,idy==j)
                ty = idy[idx].copy()
                ty[idy[idx] == i] = 1
                ty[idy[idx] == j] = -1
                # print(X[idx].shape, ty.shape)
                self.es_lst.append(demo_SVM(kname=self.kn).fit(X[idx], ty))
                self.support_ = np.hstack((self.support_, np.arange(len(idx))[idx][self.es_lst[self.loc(i,j)].support_]))
        
        self.support_ = np.unique(self.support_).astype("int")
        # print(self.support_)
        return self
                

def test_OVO_svm():
    h = 0.05
    m_svm = OVO_demo_SVM(kname='gauss')
    m_svm.fit(train_x, train_y)
    test_y_pred = m_svm.predict(test_x)
    print("OVO_svm:{:.3%}".format((test_y_pred==test_y).sum() / test_y.shape[0]))
    xx, yy = np.meshgrid(np.arange(train_x[:,0].min(), train_x[:,0].max(), h),
                      np.arange(train_x[:,1].min(), train_x[:,1].max(), h))
    plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
    Z = m_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z[Z == -1] = 2
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)
    idx = m_svm.support_
    # print(idx)
    plt.scatter(train_x[idx,0], train_x[idx,1], 
                c='none',marker="o",edgecolors='black', s=100)
    plt.title("OVR_svm")
    plt.show()

def main():
    test_sk_svm()
    test_kernel_svm()
    test_OVR_svm()
    test_OVO_svm()


if __name__ == '__main__':
    main()