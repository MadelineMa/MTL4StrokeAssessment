
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:25:36 2021
stroke or not for basic machine learning
@author: lenovo
"""

import pandas as pd 
import numpy as np 
from numpy.random import choice, seed
from random import random
from my_logistic_regression import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,auc, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.ensemble import RandomForestClassifier
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

#读取数据
fn1 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\test_ep_2019stroke.csv'
feature_column = [3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
data = pd.read_csv(fn1, usecols=feature_column)
data = data.drop(data[(data['风险评级']==0)].index)
data = data.reset_index(drop=True)
print(data['风险评级'].value_counts())

data['风险评级'] = data['风险评级'].apply(change_label)
## 切分  
#data = data.reset_index(drop=True)
#train_x, train_y, test_x, test_y = train_test_split(data, prob=0.8, random_state=20)
## 归一化
#scaler = StandardScaler()
#train_x = scaler.fit_transform(train_x)
#test_x = scaler.transform(test_x)

# 模型训练
## LR 
#clf_lr = LogisticRegression(learning_rate=0.035, max_iter=600, seed=250, mode = 'ori')
#clf_lr.fit(np.array(train_x), train_y.values)
## ghm_LR
#clf_ghmlr = LogisticRegression(learning_rate=0.035, max_iter=400, seed=250, mode = 'ghm')
#clf_ghmlr.fit(np.array(train_x), train_y.values)
## SVM 
#clf_svm=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovo', probability = True) # ovr:一对多策略
#clf_svm.fit(train_x,train_y.values) 
#
## GBDT
#clf_gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=5, subsample=1
#                                  , min_samples_split=2, min_samples_leaf=1, max_depth=3
#                                  , init=None, random_state=None, max_features=None
#                                  , verbose=0, max_leaf_nodes=None, warm_start=False
#                                  )
#clf_gbdt.fit(train_x,train_y.values)
##rf
#clf_rf = RandomForestClassifier(n_estimators = 10, max_depth=5,criterion = 'entropy', random_state = 0)
#clf_rf.fit(train_x,train_y.values)

# DNN
# 切分
dnn_train_x, dnn_train_y, dnn_validation_x, dnn_validation_y ,dnn_test_x, dnn_test_y = train_vali_test_split(data)
# 归一化
scaler = StandardScaler()
dnn_train_x = scaler.fit_transform(dnn_train_x)
dnn_test_x = scaler.transform(dnn_test_x)
dnn_validation_x =scaler.transform(dnn_validation_x)
dnn_train_y = keras.utils.to_categorical((dnn_train_y-1).values, 1)
dnn_test_y = keras.utils.to_categorical((dnn_test_y-1).values, 1)
dnn_validation_y = keras.utils.to_categorical((dnn_validation_y-1).values, 1)
# 创建一个网络模型
clf_dnn = Sequential()
clf_dnn.add(Dense(17, activation='relu', input_shape=(34,)))
clf_dnn.add(Dropout(0.05))
clf_dnn.add(Dense(17, activation='relu'))
clf_dnn.add(Dropout(0.05))
clf_dnn.add(Dense(1, activation='sigmoid'))
clf_dnn.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = clf_dnn.fit(dnn_train_x, dnn_train_y,
                    batch_size=128,
                    epochs=200,
                    verbose=1,
                    validation_data=(dnn_validation_x, dnn_validation_y))

#预测
#lr_test_pred,lr_test_pred_proba = clf_lr.predict(test_x)
#ghmlr_test_pred,ghmlr_test_pred_proba = clf_ghmlr.predict(test_x)
#svm_test_pred = clf_svm.predict(test_x)
#svm_test_pred_proba = clf_svm.predict_proba(test_x)
#gbdt_test_pred = clf_gbdt.predict(test_x)
#gbdt_test_pred_proba = clf_gbdt.predict_proba(test_x)
#rf_test_pred = clf_rf.predict(test_x)
#rf_test_pred_proba = clf_rf.predict_proba(test_x)[:,1]
dnn_test_pred = clf_dnn.predict(dnn_test_x)
dnn_test_pred_proba = clf_dnn.predict_proba(dnn_test_x)
#衡量指标 
#lr_report = classification_report(test_y.values, lr_test_pred, digits=4)
#ghmlr_report = classification_report(test_y.values, ghmlr_test_pred, digits=4)
#svm_report = classification_report(test_y.values, svm_test_pred, digits=4)
#gbdt_report = classification_report(test_y.values, gbdt_test_pred, digits=4)
#rf_report = classification_report(test_y.values, rf_test_pred, digits=4)
dnn_report = classification_report(dnn_test_y, dnn_test_pred, digits=4)
#auc
#lr_fpr, lr_tpr, thresholds = roc_curve(test_y.values, lr_test_pred_proba, pos_label=1)
#lr_auc = auc(lr_fpr, lr_tpr)
#ghmlr_fpr, ghmlr_tpr, thresholds = roc_curve(test_y.values, ghmlr_test_pred_proba, pos_label=1)
#ghmlr_auc = auc(ghmlr_fpr, ghmlr_tpr)
#svm_fpr, svm_tpr, thresholds = roc_curve(test_y.values, svm_test_pred_proba[:,1], pos_label=1)
#svm_auc = auc(svm_fpr, svm_tpr)
#gbdt_fpr, gbdt_tpr, thresholds = roc_curve(test_y.values, gbdt_test_pred_proba[:,1], pos_label=1)
#gbdt_auc = auc(gbdt_fpr, gbdt_tpr)
#rf_fpr, rf_tpr, thresholds = roc_curve(test_y.values, rf_test_pred_proba, pos_label=1)
#rf_auc = auc(rf_fpr, rf_tpr)
dnn_fpr, dnn_tpr, thresholds = roc_curve(dnn_test_y, dnn_test_pred_proba, pos_label=1)
dnn_auc = auc(dnn_fpr, dnn_tpr)


