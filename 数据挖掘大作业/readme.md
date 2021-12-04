# 数据挖掘大作业--SVM实现代码

## 需要用到的软件和库

* 语言：python 3.8.12
* 使用的库：
  * numpy  -- 矩阵运算
  * matplotlib -- 可视化
  * sklearn -- 用于对比实现SVM与sklearn.svm
  *  pyinstrument -- 用于测试运行时间



```shell
pip install pyinstrument 
pyinstrument -r html .\svm_slove_digits.py
```



## 描述

-  [demo_svm_visualization.py](demo_svm_visualization.py)  -- SVM运行及其**可视化**，运行报错需要修改图片输出的文件路径
- [multi_svm_using_numpy.py](multi_svm_using_numpy.py)  -- demo_SVM全部**实现**，支持核函数，支持多分类
- [svm_slove_digits.py](svm_slove_digits.py)  -- 用手写数字数据集测试SVM的**运行时间 **
- [kernal_split.py](kernal_split.py)  -- 核函数解决线性不可分示意图  
- [demo_svm_visualization_v2.py](demo_svm_visualization_v2.py) --核函数SVM测试和可视化

```shell
python3 *.py #直接运行即可，在ide中F5运行
```



**注：有图片生成的脚本直接运行可能会报错，请先修改图片文件的路径前缀**
