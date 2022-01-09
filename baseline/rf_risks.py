# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:59:27 2021
4-classification by Random forest
@author: M.Ma
"""

import pandas as pd 
import numpy as np 
from numpy.random import choice, seed

from my_logistic_regression import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


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
    train_data = train_data.reset_index(drop=True)
    train_data, validation_data = train_test_split0(train_data, prob=0.8, random_state=20)
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

# fn1 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\test_ep_2019stroke.csv'
fn1 = '/home/mliao/Program/Data/MLCode/Stroked/GN-data/test_ep_2019stroke.csv'
# feature_column = [3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
# data = pd.read_csv(fn1, usecols=feature_column)
data = pd.read_csv(fn1)
data.drop(data.columns[[0,1,2,6]], axis=1, inplace=True)
print(data['风险评级'].value_counts())
#去掉0
data = data[~data['风险评级'].isin([0])]
print(data['风险评级'].value_counts())
data = data.reset_index(drop=True)
# 切分
train_x, train_y, test_x, test_y = train_test_split(data, prob=0.8, random_state=20)
# 训练
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)
print(classification_report(test_y, y_pred, digits=4))
