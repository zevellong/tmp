# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 11:13:03 2021
可视化核函数的作业
@author: longz
"""

import numpy as np
import matplotlib.pyplot as plt
# import seaborn
from mpl_toolkits.mplot3d import axes3d

# seaborn.set()
prepath = "./figure/" #图片输出的目录，如果报错请自行修改

points = np.array([[3,0], [2,1], [1,2], [0,3]])
labs = np.array([0,20,20,0])

plt.figure(dpi=300)
plt.scatter(points[[0,3],0], points[[0,3],1], color="red", label="class 1")
plt.scatter(points[[1,2],0], points[[1,2],1], color="black",  label="class 2")
plt.legend()
plt.savefig(prepath + "ker1.pdf")
plt.show()


plt.figure(dpi=300)
x = np.arange(0.2,3,0.01)
plt.scatter(points[[0,3],0], points[[0,3],1], color="red", label="class 1")
plt.scatter(points[[1,2],0], points[[1,2],1], color="black",  label="class 2")
plt.plot(x, 1/x)
plt.text(0.5,2,r'$y=\frac{1}{x}$',fontdict={'size':'13','color':'b'})
plt.legend()
plt.savefig(prepath + "ker2.pdf")
plt.show()



plt.figure(dpi=2000)
h = 0.05 
rgMax, rgMin = 3, 0.2
xx, yy = np.meshgrid(np.arange(rgMin, rgMax, h), np.arange( rgMin, rgMax, h))
z = np.ones(xx.shape);
plt.figure('3D Surface')
ax = plt.axes(projection='3d',fc='whitesmoke',)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_zlabel('xy', fontsize=14)
ax.plot_surface(xx, yy, z, cstride=2, linewidth=0.5, color='orangered',  alpha=0.3)
ax.set_xticks([])
ax.set_yticks([])
ax.plot_surface(xx, yy, z*2, cstride=2, linewidth=0.5, color='g',  alpha=0.6)
ax.plot_surface(xx, yy, z*0, cstride=2, linewidth=0.5, color='g',  alpha=0.6)

ax.view_init(15, -60)
ax.text(0,1,1,r'$xy=1$',fontdict={'size':'13','color':'b'})
px = np.array([3, 2, 1, 0])
py = np.array([0, 1, 2, 3])
pz = px*py
ax.scatter(px, py, pz, c = 'black') 
plt.savefig(prepath + "ker3.pdf")
plt.show()


