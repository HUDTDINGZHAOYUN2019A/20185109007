# <center>工业蒸汽测量</center>

## 0. 成员说明

| 姓名        | 学号 | 分工职责 | 完成情况 |
| ----------- | ---- | -------- | -------- |
| 20185109007 |      |          |          |
| 2018xxxxxx  |      |          |          |
| 2018xxxxxx  |      |          |          |
| 2018xxxxxx  |      |          |          |

## 1. 前期准备

前期主要进行：

* 账号申请
* md语法学习
* 开发环境搭建
* git命令学习
* python语法学习
* 开发环境搭建
* 算法学习

### 1.1 账号创建

* [csdn账号创建](https://blog.csdn.net/NUDTDING2019)
* [github账号创建](https://github.com/HUDTDINGZHAOYUN2019A)
* [天池大赛账号创建](https://tianchi.aliyun.com/competition)

### 1.2 markdown语法学习

* [markdown教程](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#full-editor)

### 1.3 git基本命令使用

* [git教程](https://www.liaoxuefeng.com/wiki/896043488029600)

### 1.4 python基本语法学习

* [python教程](https://www.liaoxuefeng.com/wiki/1016959663602400)

### 1.5 开发环境搭建

* [python环境搭建](https://blog.csdn.net/hesong1120/article/details/78988597)

### 1.6 算法学习

* [简单算法学习](https://juejin.im/post/5b8bf792e51d4538b406f43c)

## 2.[赛题说明](https://tianchi.aliyun.com/competition/entrance/231693/information)

### 2.1 背景

火力发电的基本原理是：燃料在燃烧时加热水生成蒸汽，蒸汽压力推动汽轮机旋转，然后汽轮机带动发电机旋转，产生电能。在这一系列的能量转化中，影响发电效率的核心是锅炉的燃烧效率，即燃料燃烧加热水产生高温高压蒸汽。锅炉的燃烧效率的影响因素很多，包括锅炉的可调参数，如燃烧给量，一二次风，引风，返料风，给水水量；以及锅炉的工况，比如锅炉床温、床压，炉膛温度、压力，过热器的温度等。

### 2.2 赛题描述

经脱敏后的锅炉传感器采集的数据（采集频率是分钟级别），根据锅炉的工况，预测产生的蒸汽量。

### 2.3 数据说明

数据分成训练数据（train.txt）和测试数据（test.txt），其中字段”V0”-“V37”，这38个字段是作为特征变量，”target”作为目标变量。选手利用训练数据训练出模型，预测测试数据的目标变量，排名结果依据预测结果的MSE（mean square error）。

### 2.4 结果

最终结果为一列目标值，保存为txt文件上传到天池大赛。

## 3.数据训练

### 3.1 读取数据

数据分为两部分：

* [zhengqi_train.txt](https://github.com/HUDTDINGZHAOYUN2019A/20185109007-1/tree/master/src/zhengqi_train.txt)训练样本，用于训练模型
* [zhengqi_test.txt](https://github.com/HUDTDINGZHAOYUN2019A/20185109007-1/tree/master/src/zhengqi_test.txt)测试样本，用于结果预测

### 3.2 剔除异常值和无关特征

训练样本共有38个字段，但并非每个字段对目标值的影响都一样，可删掉影响较小的列特征。

* 计算相关系数，丢掉与target相关系数小于0.1的特征 

```
# 相关系数阀值
threshold = 0.1
#相关系数绝对值矩阵
corr_matrix = data_train.corr().abs()
drop_col=corr_matrix[corr_matrix["target"]<threshold].index
print('drop_col=',drop_col)
data_all.drop(drop_col,axis=1,inplace=True)
print('data_all.shape=',data_all.shape)
```

* 对每一列特征最大最小归一化(https://blog.csdn.net/Gamer_gyt/article/details/77761884)

```
"""归一化"""
cols_numeric=list(data_all.columns)
cols_numeric.remove("oringin")
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
scale_cols = [col for col in cols_numeric if col!='target']
print('scale_cols=',scale_cols)
data_all[scale_cols] = data_all[scale_cols].apply(scale_minmax,axis=0)
print('data_all[scale_cols].shape=',data_all[scale_cols].shape)
```

### 3.3 模型训练

采用[随机森林对模型](https://segmentfault.com/a/1190000017320801)进行训练。

训练代码为[train.py](https://github.com/HUDTDINGZHAOYUN2019A/20185109007-1/tree/master/src/train.py)。

## 4.结果预测

数据进行训练后生成[模型](https://github.com/HUDTDINGZHAOYUN2019A/20185109007-1/blob/master/src/forest_model.pkl)

利用该模型进行预测的结果代码为[predict.py](https://github.com/HUDTDINGZHAOYUN2019A/20185109007-1/tree/master/src/predict.py)。

最终预测结果：[result.txt](https://github.com/HUDTDINGZHAOYUN2019A/20185109007-1/tree/master/src/result.txt)。


## 参考文档

主要参考博客：

* [天池入门赛--蒸汽预测](https://blog.csdn.net/fanzonghao/article/details/86218579)
* [工业蒸汽量预测](https://blog.csdn.net/weixin_38404123/article/details/86645289)



