# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:07:27 2021
classification of stroke
verify ghm
@author: M.Ma
"""

import pandas as pd 
import numpy as np 
from numpy.random import choice, seed
from random import random
from my_logistic_regression import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def change_label(x):
    if x == 5:
        x = 1
    else:
        x = 0
    return x

def train_test_split(sample, prob=0.7, random_state=None):
    # Set random state.
    if random_state is not None:
        seed(random_state)
    # Split data
    n_rows, _ = sample.shape
    k = int(n_rows * prob)
    train_indexes = choice(range(n_rows), size=k, replace=False)
    test_indexes = np.array([i for i in range(n_rows) if i not in train_indexes])
    train_data = sample.loc[list(train_indexes)]
    validation_data = sample.loc[list(test_indexes)]
    validation_x, validation_y = validation_data, validation_data.pop('风险评级') 
    train_x, train_y = train_data, train_data.pop('风险评级')
    return train_x, train_y, validation_x, validation_y 

def metrics(y_test, y_pred):
    p_score = precision_score(y_test, y_pred, average='micro')
    r_score = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    #模型报告
    return p_score, r_score, f1, classification_report(y_test, y_pred, digits=4)

#测试：去掉2019卒中
fn0 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\normalStroke.csv'
data = pd.read_csv(fn0)
feature_column = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
data = data.drop(data[(data['档案年度']==2019) & (data['风险评级']==5)].index)
fn1 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\test_ep_2019stroke.csv'
data.to_csv(fn1, encoding='utf_8_sig')
feature_column = [3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
data = pd.read_csv(fn1, usecols=feature_column)
#测试完毕

# #测试：去掉2017年正在脑卒中
# fn0 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\normalStroke.csv'
# data = pd.read_csv(fn0)
# feature_column = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
# data = data.drop(data[(data['档案年度']==2017) & (data['风险评级']==5)].index)
# fn1 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\test_ep_2017stroke.csv'
# data.to_csv(fn1, encoding='utf_8_sig')
# feature_column = [3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
# data = pd.read_csv(fn1, usecols=feature_column)
# #测试完毕


# fn = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\normalStroke.csv'
# feature_column = [2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
# data = pd.read_csv(fn, usecols=feature_column)
print(data['风险评级'].value_counts())
#测试：和stroke分不开的是l,m,h的哪一部分
data = data[~data['风险评级'].isin([0])]
data = data[~data['风险评级'].isin([2])]
data = data[~data['风险评级'].isin([1])]
print(data['风险评级'].value_counts())
fn2 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\test_ls.csv'
data.to_csv(fn2, encoding='utf_8_sig')
# feature_column = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
data = pd.read_csv(fn2)
data = data.drop(columns=['Unnamed: 0'])
print(data['风险评级'].value_counts())
#测试完毕

#测试：去掉一些特征
# data = data.drop(columns=['冠心病'])
# data = data.drop(columns=['高血压'])
# data = data.drop(columns=['糖尿病'])
# data = data.drop(columns=['身高'])
# data = data.drop(columns=['体重'])
# data = data.drop(columns=['食用蔬菜'])
#测试完毕

data['风险评级'] = data['风险评级'].apply(change_label)
# 切分  
data = data.reset_index(drop=True)
train_x, train_y, test_x, test_y = train_test_split(data, prob=0.8, random_state=20)
# 归一化
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

#模型训练
clf1 = LogisticRegression(learning_rate=0.035, max_iter=600, seed=250, mode = 'ori')
clf2 = LogisticRegression(learning_rate=0.035, max_iter=400, seed=250, mode = 'ghm')
clf1.fit(np.array(train_x), train_y.values)
clf2.fit(np.array(train_x), train_y.values)

#预测
y1_test_pred,y1_test_pred_proba = clf1.predict(test_x)
y2_test_pred,y2_test_pred_proba = clf2.predict(test_x)
#衡量指标
p_score1, r_score1, f11, cf_report1 = metrics(test_y.values, y1_test_pred)
print(cf_report1)
p_score2, r_score2, f12, cf_report2 = metrics(test_y.values, y2_test_pred)
print(cf_report2)


#特征重要性画图
# plt.figure()
# name=['gender','age','nation','marital status','education','retire','smoking','smokeing years','drinking','exercise','taste','meat&vagetables','eatting vagetables','eatting fruit','stroke','coronary','Hypertension','diabetes','height','weight','BMI','left systolic','left diastolic','right systolic','right diastolic','heart rhythm','pulse','fasting blood glucose','glycosylated hemoglobin','triglycerides','total cholesterol ','low density lipoprotein cholesterol','high density lipoprotein cholesterol','homocysteineglobin']
# # name=['gender','age','nation','marital status','education','retire','smoking','smokeing years','drinking','exercise','taste','meat&vagetables','eatting fruit','stroke','BMI','left systolic','left diastolic','right systolic','right diastolic','heart rhythm','pulse','fasting blood glucose','glycosylated hemoglobin','triglycerides','total cholesterol ','low density lipoprotein cholesterol','high density lipoprotein cholesterol','homocysteineglobin']

# w=clf2.w
# markerline, stemlines, baseline =plt.stem(name,w,linefmt='-',markerfmt='o',basefmt='--')
# plt.xticks(rotation=90)
# plt.setp(markerline, color='k')
# plt.legend()
# plt.show()

