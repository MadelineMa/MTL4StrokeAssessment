# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:59:27 2021
4-classification by Random forest
@author: M.Ma
"""

import pandas as pd 
import numpy as np 
import prettytable as pt
from numpy.random import choice, seed

from my_logistic_regression import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score #, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

def statistics(X, alpha=0.95):
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    SE = sig/np.sqrt(X.shape[0])
    CI = stats.norm.interval(alpha, loc=mu, scale=SE)
    return mu, sig, CI[0], CI[1]

ntest = 10
PScr = np.zeros((ntest,4))
RScr = np.zeros((ntest,4))
FScr = np.zeros((ntest,4))
# 训练

fn1 = '/home/mliao/Program/Data/MLCode/Stroked/GN-data/dnn_train.csv'
data = pd.read_csv(fn1)
data.pop('Apx')
train_x, train_y = data, data.pop('RR')
fn2 = '/home/mliao/Program/Data/MLCode/Stroked/GN-data/dnn_test.csv'
data = pd.read_csv(fn2)
data.pop('Apx')
test_x, test_y = data, data.pop('RR')
# y_pred = classifier.predict(test_x)
for i in range(ntest):
    classifier = RandomForestClassifier(n_estimators = 10+i, criterion = 'entropy', random_state = 0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    PScr[i] = precision_score(test_y, y_pred, average=None)
    RScr[i] = recall_score(test_y, y_pred, average=None)
    FScr[i] = f1_score(test_y, y_pred, average=None)

np.savez('./npz/rf.npz', PScr=PScr, RScr=RScr, FScr=FScr)
mP, sP, PCIL, PCIR = statistics(PScr)
mR, sR, RCIL, RCIR = statistics(RScr)
mF, sF, FCIL, FCIR = statistics(FScr)

tb = pt.PrettyTable()
ntable = ['Low', 'Medium', 'High', 'Attact']
for i in range(len(ntable)):
    tb.field_names = ["RF", "precision", "std", "CIL", "CIR"]
    tb.add_row([ntable[i], mP[i], sP[i], PCIL[i], PCIR[i]])
print(tb)
tb = pt.PrettyTable()
for i in range(len(ntable)):
    tb.field_names = ["RF", "Recall", "std", "CIL", "CIR"]
    tb.add_row([ntable[i], mR[i], sR[i], RCIL[i], RCIR[i]])
print(tb)
tb = pt.PrettyTable()
for i in range(len(ntable)):
    tb.field_names = ["RF", "f1-score", "Fstd", "FCIL", "FCIR"] #, "#iteration", "Istd", "ICIL", "ICIR"]
    tb.add_row([ntable[i], mF[i], sF[i], FCIL[i], FCIR[i]])
print(tb)

# +--------+--------------------+----------------------+--------------------+--------------------+
# |   RF   |     precision      |         std          |        CIL         |        CIR         |
# +--------+--------------------+----------------------+--------------------+--------------------+
# |  Low   | 0.8362131909541256 | 0.003204641870865766 | 0.8342269698422478 | 0.8381994120660035 |
# | Medium | 0.8522889809414529 | 0.008502722361782598 | 0.8470190358605552 | 0.8575589260223506 |
# |  High  | 0.8576691285030911 | 0.010756823813684601 | 0.851002102539994  | 0.8643361544661882 |
# | Attact | 0.9246305296885508 | 0.006162822488573621 | 0.920810842925158  | 0.9284502164519435 |
# +--------+--------------------+----------------------+--------------------+--------------------+
# +--------+--------------------+----------------------+--------------------+--------------------+
# |   RF   |       Recall       |         std          |        CIL         |        CIR         |
# +--------+--------------------+----------------------+--------------------+--------------------+
# |  Low   | 0.9679744525547447 | 0.004170217110228625 | 0.9653897727061935 | 0.9705591324032958 |
# | Medium | 0.8629965947786606 | 0.005985832859675457 | 0.8592866053080284 | 0.8667065842492929 |
# |  High  | 0.7708487084870849 | 0.011300727548086968 | 0.7638445736913533 | 0.7778528432828165 |
# | Attact | 0.6010344827586207 | 0.009512492568368077 | 0.5951386871199121 | 0.6069302783973293 |
# +--------+--------------------+----------------------+--------------------+--------------------+
# +--------+--------------------+-----------------------+--------------------+--------------------+
# |   RF   |      f1-score      |          Fstd         |        FCIL        |        FCIR        |
# +--------+--------------------+-----------------------+--------------------+--------------------+
# |  Low   | 0.8972800984615382 | 0.0033344195328740894 | 0.8952134417994435 | 0.8993467551236329 |
# | Medium | 0.8576052738087373 | 0.0071383164303316345 | 0.8531809807461997 | 0.8620295668712749 |
# |  High  | 0.8119337986265199 |  0.010732923065819343 | 0.8052815862282181 | 0.8185860110248218 |
# | Attact | 0.7284931192651731 |  0.008617165592742736 | 0.7231522428382454 | 0.7338339956921008 |
# +--------+--------------------+-----------------------+--------------------+--------------------+