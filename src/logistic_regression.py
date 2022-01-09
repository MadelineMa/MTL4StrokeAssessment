import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
# from strokeDataset import StrokedDataset
from strokeDataset import BinaryDataset
from sklearn.metrics import classification_report

class Stroke_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(34,1)
    def forward(self, x):
        return torch.sigmoid(self.lin(x)) # change to F.sigmoid(self.lin(x))

def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)

def fit(epochs, model, loss_func, opt, train_dl, test_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)
        # if epoch % 10 == 0:
            # print(epoch, val_loss)
def calcAUC_byRocArea(y_true=None, y_pred_proba=None ):
    if y_true is None or y_pred_proba is None:
        ValueError("Value should be givein explicitly")
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

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = '../GN-data/train.csv'
        test_file = '../GN-data/test.csv'
    else:
        train_file = '..\\GN-data\\train.csv'
        test_file = '..\\GN-data\\test.csv'
    lr = 0.01
    bs = 100
    epochs = 100
    # train_ds = StrokedDataset(train_file)
    # test_ds = StrokedDataset(test_file, train_ds.scaler)
    train_ds = BinaryDataset(train_file)
    test_ds = BinaryDataset(test_file, train_ds.scaler)
    # data, label = myDataset[10]
    # print('data: ', data)
    # print('label: ', label)
    # print('length: ', len(myDataset))

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    model = Stroke_Model()
    # loss_func = F.cross_entropy
    # loss_func = F.binary_cross_entropy
    loss_func = nn.BCELoss()
    opt = optim.SGD(model.parameters(), lr=lr,)
    fit(epochs, model, loss_func, opt, train_dl, test_dl)
    y_prob = model(test_ds.data)
    y_prob = y_prob.squeeze().tolist()
    y_pred = np.array([0 if y_prob[i] < 0.5 else 1 for i in range(len(y_prob))])
    y_test = test_ds.labels.tolist()
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    AUC = calcAUC_byRocArea(y_test, y_prob)