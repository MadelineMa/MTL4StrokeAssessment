# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:16:31 2020

@author: M.Ma
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
class LogisticRegression(object):

    def __init__(self, learning_rate=0.1, max_iter=100, seed=None, mode = 'ori'):
        self.seed = seed
        self.lr = learning_rate
        self.max_iter = max_iter
        self.mode = mode

    def fit(self, x, y):
        np.random.seed(self.seed)
        self.w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1])
        self.b = np.random.normal(loc=0.0, scale=1.0)
        self.x = x
        self.y = y
        loss=[]
        auc=[]
        my_recall_score=[]
        my_precision_score =[]
        for i in range(self.max_iter):
            self._update_step()
            loss.append(self.loss())
            auc.append(self.calcAUC_byRocArea())
            my_precision_score.append(self.cal_precision_score())
            my_recall_score.append(self.cal_recall_score())
#            print('loss: \t{}'.format(self.loss()))
            # print('score: \t{}'.format(self.score()))
            # print('w: \t{}'.format(self.w))
            # print('b: \t{}'.format(self.b))
        plt.figure()
        plt.plot(np.arange(self.max_iter),np.array(loss),c='y')
        plt.title('loss')
        plt.show()
        
        plt.figure()
        plt.plot(np.arange(self.max_iter),np.array(auc),c='b')
        plt.title('auc')        
        plt.show()
        
        plt.figure()
        plt.plot(np.arange(self.max_iter),np.array(my_recall_score),c='b')
        plt.title('recall')        
        plt.show()
        
        plt.figure()
        plt.plot(np.arange(self.max_iter),np.array(my_precision_score),c='b')
        plt.title('precision')        
        plt.show()
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _f(self, x, w, b):
        z = x.dot(w) + b
        return self._sigmoid(z)

    def predict_proba(self, x=None):
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred

    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred_proba = self._f(x, self.w, self.b)
        y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])
        return y_pred, y_pred_proba

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred, y_pred_proba = self.predict()   
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

    def loss(self, y_true=None, y_pred_proba=None):
        if y_true is None or y_pred_proba is None:
            y_true = self.y
            y_pred_proba = self.predict_proba()
        #print('y_true:',y_true,'y_pred_proba:',y_pred_proba)
        return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))
    
    #for 正样本 1 
    def cal_recall_score(self,  y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred,y_pred_proba = self.predict()
        return recall_score(y_true,y_pred)
        
    def cal_precision_score(self,  y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred,y_pred_proba = self.predict()
        return precision_score(y_true,y_pred)
        
    def calcAUC_byRocArea(self,  y_true=None, y_pred_proba=None ):
        if y_true is None or y_pred_proba is None:
            y_true = self.y
            y_pred_proba = self.predict_proba() 
        ###initialize
        P = 0
        N = 0
        for i in y_true:
            if (i == 1):
                P += 1
            else:
                N += 1
        TP = 0
        FP = 0
        TPR_last = 0
        FPR_last = 0
        AUC = 0
        pair = zip(y_pred_proba, y_true)
        pair = sorted(pair, key=lambda x:x[0], reverse=True)
        i = 0
        while i < len(pair):
            if (pair[i][1] == 1):
                TP += 1
            else:
                FP += 1
            ### maybe have the same probs
            while (i + 1 < len(pair) and pair[i][0] == pair[i+1][0]):
                i += 1
                if (pair[i][1] == 1):
                    TP += 1
                else:
                    FP += 1
            TPR = TP / P
            FPR = FP / (N+0.00001)
            AUC += 0.5 * (TPR + TPR_last) * (FPR - FPR_last)
            TPR_last = TPR
            FPR_last = FPR
            i += 1
        return AUC

    
    def GHM(self,  y_pred):
        batch_size =len(self.y)
        g = abs(y_pred-self.y)
#        print('g:',g)
#        print('y_pred:',y_pred)
        hist, bin_edges = np.histogram(g, bins = 10, range=[0,1])
        GD = hist/0.1
        beta = []
        for g_i in g:
            
            GD_gi = GD[np.where(g_i <= bin_edges)[0][0]-1]
            if GD_gi != 0:
                beta_i = batch_size/GD_gi
            else:
                beta_i = batch_size/0.1
            beta.append(beta_i)
        return beta 
    
    
    def _calc_gradient(self):
        y_pred,y_pred_proba = self.predict()
        if self.mode == 'ori':
            #original method
            d_w = (y_pred - self.y).dot(self.x) / len(self.y)
            d_b = np.mean(y_pred - self.y)
        elif self.mode == 'ghm':
            #--add beta
            
            beta = self.GHM(y_pred_proba)
#            plt.plot(beta,'r*')
#            plt.show()
            d_w = (beta * (y_pred - self.y)).dot(self.x) / len(self.y)
            d_b =  np.mean(beta *(y_pred - self.y))
        return d_w, d_b

    def _update_step(self):
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b

