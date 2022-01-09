# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 23:01:19 2021
classify of risk
@author: lenovo
"""
import pandas as pd 
import numpy as np 
from numpy.random import choice, seed

from my_logistic_regression import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def change_label1(x):
    if x == 1:
        x = 0
    else:
        x = 1
    return x

def change_label2(x):
    if x == 2:
        x = 0
    else:
        x = 1
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


def train_test_split0(sample, prob=0.7, random_state=None):
    # Set random state.
    if random_state is not None:
        seed(random_state)
    # Split data
    n_rows, _ = sample.shape
    k = int(n_rows * prob)
    train_indexes = choice(range(n_rows), size=k, replace=False)
    test_indexes = np.array([i for i in range(n_rows) if i not in train_indexes])
    train_data = sample.loc[list(train_indexes)]
    test_data = sample.loc[list(test_indexes)]
    return train_data, test_data

def train_vali_test_split(data):
    train_data, test_data = train_test_split0(data, prob=0.85, random_state=20)
    train_data, validation_data = train_test_split0(data, prob=0.8, random_state=20)
    validation_x, validation_y = validation_data, validation_data.pop('风险评级') 
    train_x, train_y = train_data, train_data.pop('风险评级')
    test_x, test_y = test_data, test_data.pop('风险评级')
    return train_x, train_y, validation_x, validation_y, test_x, test_y
    

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
print(data['风险评级'].value_counts())
#去掉stroking 和 0
data = data[~data['风险评级'].isin([0])]
data = data[~data['风险评级'].isin([5])]
print(data['风险评级'].value_counts())
data = data.reset_index(drop=True)
# 切分  
train_x, train_y, test_x, test_y = train_test_split(data, prob=0.8, random_state=20)
# 归一化
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# # 测试随机森林
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(train_x, train_y)
# y_pred = classifier.predict(test_x)
# classification_report(test_y, y_pred, digits=4)
# # 测试完毕




#低危:0 vs 中+高危:1
train_y1 = train_y.copy()
train_y1 = train_y1.apply(change_label1)
# clf1 = LogisticRegression(learning_rate=0.035, max_iter=600, seed=250, mode = 'ori')
clf1 = LogisticRegression(learning_rate=0.035, max_iter=400, seed=250, mode = 'ghm')
clf1.fit(np.array(train_x), train_y1.values)

#中:0, 高危:1
#删除低危
train_y = train_y.reset_index(drop=True)
index_train = train_y[train_y == 1].index
train_y = train_y.drop(index_train)
train_x = np.delete(train_x, list(index_train), axis=0)
# label变换
train_y2 = train_y.apply(change_label2)
# clf2 = LogisticRegression(learning_rate=0.035, max_iter=600, seed=250, mode = 'ori')
clf2 = LogisticRegression(learning_rate=0.035, max_iter=400, seed=250, mode = 'ghm')
clf2.fit(np.array(train_x), train_y2.values)

#预测
y1_test_pred,y1_test_pred_proba = clf1.predict(test_x)
ml_index = np.where(y1_test_pred==1)[0]
test_x2 = test_x.copy()
test_y2 = test_y.copy()
test_x2 = test_x2[ml_index]
test_y2 = test_y2.values[ml_index]
y2_test_pred,y2_test_pred_proba = clf2.predict(test_x2)

#y_test label调整
l_index = np.where(y1_test_pred == 0)[0]
m_in_index = np.where(y2_test_pred == 0)[0]
m_index = ml_index[m_in_index]
h_in_index = np.where(y2_test_pred == 1)[0]
h_index = ml_index[h_in_index]

y1_test_pred[l_index] = 1
y1_test_pred[m_index] = 2
y1_test_pred[h_index] = 3
#衡量指标
p_score, r_score, f1, cf_report = metrics(test_y.values, y1_test_pred)
print(cf_report)












