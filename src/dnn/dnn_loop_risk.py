import numpy as np
import sys
from numpy.core.arrayprint import _leading_trailing
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import prettytable as pt
import os
if os.name == 'posix':
    sys.path.append("../")
else:
    sys.path.append("..\\")
from strokeDataset import DnnDataset
from sklearn.metrics import precision_score, recall_score, f1_score #, classification_report
from early_stopping import EarlyStopping
from utils import statistics

epsilon = torch.tensor(sys.float_info.epsilon)

dnnDir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'posix':
   ptDir = os.path.join(dnnDir, "pt/")
   npzDir = os.path.join(dnnDir, "npz/")
else:
   ptDir = os.path.join(dnnDir, "pt\\")
   npzDir = op.path.join(dnnDir, "npz\\")

class Stroke_DNN_Model(nn.Module):
    def __init__(self, num_dims):
        super().__init__()
        for i in range(1, 3):
            setattr(self, 'lin'+str(i), 
                    nn.Linear(num_dims[i-1], num_dims[i]))
            setattr(self, 'relu'+str(i),
                    nn.ReLU())
            setattr(self, 'dropout'+str(i),
                    nn.Dropout(p=0.2))
    def forward(self, x):
        for i in range(1, 3):
            x = getattr(self, 'lin'+str(i))(x)
            x = getattr(self, 'relu'+str(i))(x)
            x = getattr(self, 'dropout'+str(i))(x)
        return x

def loss_batch(model, loss_func, x, y, opt=None):
    p_risk = model(x)
    loss = loss_func(p_risk, y)  
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, early_stopping=None):
    loss_v = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        loss_v.append(val_loss)
        # print(epoch, val_loss)

        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    idx = np.argmin(loss_v)
    print('minimum: ')
    print('epoch: {}, loss: {}'.format(idx, loss_v[idx]))
    # plt.plot(np.arange(len(loss_v)), np.array(loss_v),c='y')
    # plt.plot(idx, loss_v[idx], marker='o')
    # plt.title('loss-dnn')
    # plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))      
    return idx

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = '../../GN-data/dnn_train.csv'
        valid_file = '../../GN-data/dnn_valid.csv'
        test_file = '../../GN-data/dnn_test.csv'
    #    train_file = '../QI/data/qi_train.csv'
    #    valid_file = '../QI/data/qi_valid.csv'
    #    test_file = '../QI/data/qi_test.csv'
    else:
    #    train_file = '..\\..\\GN-data\\dnn_train.csv'
    #    valid_file = '..\\..\\GN-data\\dnn_valid.csv'
    #    test_file = '..\\..\\GN-data\\dnn_test.csv'
        train_file = '..\\QI\\data\\qi_train.csv'
        valid_file = '..\\QI\\data\\qi_valid.csv'
        test_file = '..\\QI\\data\\qi_test.csv'
    lr = 0.01
    bs = 100
    epochs = 400
    train_ds = DnnDataset(train_file)
    valid_ds = DnnDataset(valid_file, train_ds.scaler)
    test_ds = DnnDataset(test_file, train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    num_dims = [34, 17, 4]

    npzFn = os.path.join(npzDir, "dnn_risk.npz")
    try:
        with np.load(npzFn) as f:
            PScr, RScr, FScr, nIter = f['PScr'], f['RScr'], f['FScr'], f['nIter']
    except FileNotFoundError:
        ntest = 50
        patience = 20
        PScr = np.zeros((ntest,4))
        RScr = np.zeros((ntest,4))
        FScr = np.zeros((ntest,4))
        nIter = np.zeros(ntest)

        for i in range(ntest):
            print('Computing {}/{}...'.format(i+1, ntest))
            model = Stroke_DNN_Model(num_dims)
            loss_func = nn.CrossEntropyLoss()
            # opt = optim.RMSprop(model.parameters(), lr=lr,)
            opt = optim.Adam(model.parameters(), lr=lr,)

            early_stopping = EarlyStopping(patience, verbose=False)

            niter = fit(epochs, model, loss_func, opt, train_dl, valid_dl, early_stopping)
            print('Trained with {} iterations...'.format(niter))
            y_prob = model(test_ds.data)
            y_pred = np.argmax(y_prob.tolist(), axis=1)
            y_test = test_ds.labels.tolist()
            # report = classification_report(y_test, y_pred, digits=4)
            # print(report)
            PScr[i] = precision_score(y_test, y_pred, average=None)
            RScr[i] = recall_score(y_test, y_pred, average=None)
            FScr[i] = f1_score(y_test, y_pred, average=None)
            nIter[i] = niter

        # PScr = np.delete(PScr, 0, 0) # MLiao: should delete "Precision is ill-defined and being set to 0.0 in labels" by hands...
        delIdx = np.where(abs(PScr) < 1e-5)[0]
        print('remove {} records...'.format(len(delIdx)))
        PScr = np.delete(PScr, delIdx, 0)
        RScr = np.delete(RScr, delIdx, 0)
        FScr = np.delete(FScr, delIdx, 0)
        nIter = np.delete(nIter, delIdx, 0)
        np.savez(npzFn, PScr=PScr, RScr=RScr, FScr=FScr, nIter=nIter)

    mP, sP, PCIL, PCIR = statistics(PScr)
    mR, sR, RCIL, RCIR = statistics(RScr)
    mF, sF, FCIL, FCIR = statistics(FScr)
    mI, sI, ICIL, ICIR = statistics(nIter)

    tb = pt.PrettyTable()
    ntable = ['Low', 'Medium', 'High', 'Attact']
    for i in range(len(ntable)):
        tb.field_names = ["DNN-CE", "precision", "std", "CIL", "CIR"]
        tb.add_row([ntable[i], mP[i], sP[i], PCIL[i], PCIR[i]])
    print(tb)
    tb = pt.PrettyTable()
    for i in range(len(ntable)):
        tb.field_names = ["DNN-CE", "Recall", "std", "CIL", "CIR"]
        tb.add_row([ntable[i], mR[i], sR[i], RCIL[i], RCIR[i]])
    print(tb)
    tb = pt.PrettyTable()
    for i in range(len(ntable)):
        tb.field_names = ["DNN-CE", "f1-score", "Fstd", "FCIL", "FCIR"] #, "#iteration", "Istd", "ICIL", "ICIR"]
        tb.add_row([ntable[i], mF[i], sF[i], FCIL[i], FCIR[i]])
    print(tb)
    tb = pt.PrettyTable()
    tb.field_names = ["DNN-CE", "#iteration", "Istd", "ICIL", "ICIR"]
    tb.add_row(['iteration', mI, sI, ICIL, ICIR])
    print(tb)