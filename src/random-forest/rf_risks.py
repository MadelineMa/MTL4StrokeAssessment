# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:59:27 2021
4-classification by Random forest
@author: M.Ma
"""

import pandas as pd 
import numpy as np 
from numpy.random import choice, seed
import os
import shap
# shap.initjs()

# from my_logistic_regression import LogisticRegression
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
    validation_x, validation_y = validation_data, validation_data.pop('RR') 
    train_x, train_y = train_data, train_data.pop('RR')
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
    validation_x, validation_y = validation_data, validation_data.pop('RR') 
    train_x, train_y = train_data, train_data.pop('RR')
    test_x, test_y = test_data, test_data.pop('RR')
    return train_x, train_y, validation_x, validation_y, test_x, test_y
    

def metrics(y_test, y_pred):
    p_score = precision_score(y_test, y_pred, average='micro')
    r_score = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    #模型报告
    return p_score, r_score, f1, classification_report(y_test, y_pred, digits=4)

if __name__ == '__main__':
    if os.name == 'posix':
        # fn1 = './test_ep_2019stroke.csv'
        fn1 = '../../GN-data/normalStroke.csv'
    else:
        # fn1 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\test_ep_2019stroke.csv'
        fn1 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\normalStroke.csv'
    # feature_column = [3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
    # data = pd.read_csv(fn1, usecols=feature_column)
    data = pd.read_csv(fn1)
    data = data.drop(data[(data['档案年度']==2019) & (data['风险评级']==5)].index)
    data = data.drop(data[(data['风险评级']==0)].index)
    # data.drop('档案年度', axis=1, inplace=True)
    data.drop(['身份证号', '档案年度'], axis=1, inplace=True)
    print(data['风险评级'].value_counts())

    data = data.reset_index(drop=True)
    data = data.rename(columns={'性别': 'Gd', '建档年龄': 'FA', '民族': 'Nat', '婚姻状况': 'MS', '受教育程度': 'Edu',
                            '是否退休': 'Ret', '风险评级': 'RR', '卒中': 'Apx', '吸烟': 'Sm', '吸烟年限': 'YS',
                            '饮酒': 'Drk', '缺乏运动': 'Exs', '口味': 'Flv', '荤素': 'MV', '食用蔬菜': 'VC',
                            '食用水果': 'FC', '脑卒中': 'HA', '冠心病': 'HCAD', '高血压': 'HEH', '糖尿病': 'HDM',
                            '身高': 'Ht', '体重': 'Wt', '左侧收缩压': 'LSBP', '左侧舒张压': 'LDBP',
                            '右侧收缩压': 'RSBP', '右侧舒张压': 'RDBP', '心律': 'Rhm', '脉搏': 'Pls',
                            '空腹血糖': 'FBG', '糖化血红蛋白': 'HbA1c', '甘油三脂': 'TG', '总胆固醇': 'TC',
                            '低密度脂蛋白胆固醇': 'LDL-C', '高密度脂蛋白胆固醇': 'HDL-C', '同型半胱氨酸': 'Hcy'})
    # 切分
    train_x, train_y, test_x, test_y = train_test_split(data, prob=0.8, random_state=20)
    # 训练
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    print(classification_report(test_y, y_pred, digits=4))

    # explainer = shap.TreeExplainer(classifier)
    # shap_interaction_values = explainer.shap_interaction_values(test_x[0:100])
    # shap.summary_plot(shap_interaction_values, test_x)
    # shap.summary_plot(shap_interaction_values[1], test_x[0:100], max_display=10)