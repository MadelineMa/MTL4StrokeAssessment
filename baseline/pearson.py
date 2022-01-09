# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:34:24 2021

@author: lenovo
"""

import pandas as pd 
import numpy as np 
from numpy.random import choice, seed

from my_logistic_regression import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import pearsonr

fn1 = 'D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\test_ep_2019stroke.csv'
feature_column = [3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
data = pd.read_csv(fn1, usecols=feature_column)
print(data['风险评级'].value_counts())
risk = np.array(data['风险评级'])
risk_v  = []
for i in range(0, len(risk)):
    if risk[i] == 1:
        risk_v.append([1,0,0,0])
    elif risk[i] ==2:
        risk_v.append([0,1,0,0])
    elif risk[i] == 3:
        risk_v.append([0,0,1,0])
    elif risk[i] == 5:
        risk_v.append([0,0,0,1])
risk_array = np.array(risk_v)
cor_lm = pearsonr(risk_array[:,0],risk_array[:,1])
cor_mh = pearsonr(risk_array[:,1],risk_array[:,2])
cor_hs = pearsonr(risk_array[:,2],risk_array[:,3])

cor_lh = pearsonr(risk_array[:,0],risk_array[:,2])
cor_ls = pearsonr(risk_array[:,0],risk_array[:,3])
cor_ms = pearsonr(risk_array[:,1],risk_array[:,3])

x= pd.Series(risk_array[:,0])
y= pd.Series(risk_array[:,1])
lm_r = x.corr(y,method="kendall") 