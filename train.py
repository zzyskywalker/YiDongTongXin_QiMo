# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:46:28 2020

@author: 33578
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score# roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

pd_data=pd.read_csv("label/train.csv")#经过清洗、时频域特征提取、合并、标签之之后训练
y=np.array(pd_data.loc[:,"label"])
x=np.array(pd_data.drop(columns={"label"}, inplace=False))
print(x.shape)
print(y.shape)

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.1,random_state=123)#划分测试集和训练集
#随机森林分类器
clf = RandomForestClassifier(n_estimators=400,
                             oob_score=True,
                             max_features=1,
                             min_samples_split=5,
                             ).fit(train_x, train_y)
#对测试集进行分析
predict = clf.predict(test_x)
#计算分类指标
accuracy = accuracy_score(test_y, predict)
precision = precision_score(test_y, predict, average="weighted")
recall = recall_score(test_y, predict,average="weighted")
f1= f1_score(test_y, predict,average="weighted")
print("accuracy: ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("f1_score: ",f1)
joblib.dump(clf,"model")