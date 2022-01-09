import numpy as np
import sys
from numpy.core.defchararray import title
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import os
if os.name == 'posix':
    sys.path.append("../")
else:
    sys.path.append("..\\")
from strokeDataset import BinaryDataset
from early_stopping import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, auc, roc_curve
epsilon = torch.tensor(sys.float_info.epsilon)

class Stroke_DNN_Model(nn.Module):
    def __init__(self, num_dims, droprate=0.2):
        super().__init__()
        self.num_dims = num_dims
        # num_dims = [34, 16, 8, 1]
        for i in range(1, len(num_dims)-1):
            setattr(self, 'lin'+str(i), 
                    nn.Linear(num_dims[i-1], num_dims[i]))
            setattr(self, 'relu'+str(i),
                    nn.ReLU())
            setattr(self, 'dropout'+str(i),
                    nn.Dropout(p=droprate))
        self.out = nn.Linear(num_dims[-2], num_dims[-1])
    def forward(self, x):
        for i in range(1, len(self.num_dims)-1):
            x = getattr(self, 'lin'+str(i))(x)
            x = getattr(self, 'relu'+str(i))(x)
            x = getattr(self, 'dropout'+str(i))(x)
        return torch.sigmoid(self.out(x))

def loss_batch(model, loss_func, x, y, opt=None):
    p_a = model(x)
    loss = loss_func(p_a, y)  #+ 10*lhs.sum()/len(x)#+ lhm.sum()/len(x) #+ lhl.sum()/len(x) + lhm.sum()/len(x)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)#, lhs.sum()/len(x)

def calcAUC_byRocArea(y_true, y_pred_proba):
    if y_true is None or y_pred_proba is None:
        ValueError('labels and predictions should be given explicitly')
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
        FPR = FP / N
        AUC += 0.5 * (TPR + TPR_last) * (FPR - FPR_last)
        TPR_last = TPR
        FPR_last = FPR
        i += 1
    return AUC

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, valid_ds, early_stopping=None):
    loss_v = []
    auc_v = []
    y_label = np.array(valid_ds.labels)
    x_valid = valid_ds.data
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        loss_v.append(val_loss)
        y_model = model(x_valid)
        auc = calcAUC_byRocArea(y_label, torch.detach(y_model).numpy())
        auc_v.append(auc)

        # print(epoch, val_loss)


        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    idx = np.argmin(loss_v)
    print('minimum (loss): ', f'epoch: {idx}, loss: {loss_v[idx]}')
    print('auc: ', auc_v[idx])
    _, axes = plt.subplots(2,1, figsize=(6, 6))
    colors = ['y', 'b']
    indices = [idx, idx]
    title = ['loss', 'auc']
    plt_data = [np.array(loss_v), np.array(auc_v)]
    for ax, c, t, d in zip(axes.flatten(), colors, title, plt_data):
        ax.plot(np.arange(len(d)), d, c=c)
        ax.plot(idx, d[idx], marker='o')
        ax.set_title(t)

    plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))      

def get_mean_index(y_prob, ev, col, shift):
    y = np.array(y_prob[:, col])
    return np.argmin(abs(y-(ev[col]+shift)/2)) 

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = '../../GN-data/dnn_train.csv'
        valid_file = '../../GN-data/dnn_valid.csv'
        test_file = '../../GN-data/dnn_test.csv'
    else:
        train_file = '..\\..\\GN-data\\dnn_train_sf.csv'
        valid_file = '..\\..\\GN-data\\dnn_valid_sf.csv'
        test_file = '..\\..\\GN-data\\dnn_test_sf.csv'
    lr = 0.01
    bs = 100
    epochs = 400
    train_ds = BinaryDataset(train_file)
    valid_ds = BinaryDataset(valid_file, train_ds.scaler)
    test_ds = BinaryDataset(test_file, train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    num_dims = [34, 17, 1]
    model = Stroke_DNN_Model(num_dims)
    loss_func = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=lr,)

    patience = 40	
    early_stopping = EarlyStopping(patience, verbose=False)

    fit(epochs, model, loss_func, opt, train_dl, valid_dl, valid_ds, early_stopping)
    y_prob = model(valid_ds.data)
    y_prob = y_prob.squeeze().tolist()
    y_pred = np.array([0 if y_prob[i] < 0.5 else 1 for i in range(len(y_prob))])
    y_test = valid_ds.labels.tolist()
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    dnn_fpr, dnn_tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    dnn_auc = auc(dnn_fpr, dnn_tpr)
    print('AUC', dnn_auc)
