# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:11:38 2021

@author: lenovo
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
#测试DNN
# 切分
train_x, train_y, validation_x, validation_y ,test_x, test_y = train_vali_test_split(data)
# 归一化
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
validation_x =scaler.transform(validation_x)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
train_y = keras.utils.to_categorical((train_y-1).values, 3)
test_y = keras.utils.to_categorical((test_y-1).values, 3)
validation_y = keras.utils.to_categorical((validation_y-1).values, 3)
# 创建一个网络模型
model = Sequential()
# 创建输入层 512代表的是输出维度为512，也就是第二层神经元有512个，输入维度为(784,)，激活函数为Relu
model.add(Dense(17, activation='relu', input_shape=(34,)))
model.add(Dropout(0.2))

# 创建layer2，然后向下层输出的空间维度为512
model.add(Dense(17, activation='relu'))
model.add(Dropout(0.2))

# 输出层,因为只有10个数字，所以输出空间维度为10，激活函数为softmax。
model.add(Dense(3, activation='softmax'))
from keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(train_x, train_y,
                    batch_size=128,
                    epochs=400,
                    verbose=1,
                    validation_data=(validation_x, validation_y))
#评估
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
pre=model.predict(test_x)
y_pred = np.argmax(pre, axis=1)
test_y = np.argmax(test_y,axis=1)
print(classification_report(test_y, y_pred, digits=4))
# 绘制训练过程中训练集和测试集合的准确率值
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 绘制训练过程中训练集和测试集合的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()








