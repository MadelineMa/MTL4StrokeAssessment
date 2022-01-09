import numpy as np
import sys
from numpy.core.defchararray import title
import torch
import prettytable as pt
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
from utils import statistics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
epsilon = torch.tensor(sys.float_info.epsilon)

dnnDir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'posix':
   ptDir = os.path.join(dnnDir, "pt/")
   npzDir = os.path.join(dnnDir, "npz/")
else:
   ptDir = os.path.join(dnnDir, "pt\\")
   npzDir = op.path.join(dnnDir, "npz\\")

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
    loss = loss_func(p_a, y) 

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, valid_ds, early_stopping=None):
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

        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    idx = np.argmin(loss_v)
    print('minimum: ')
    print('epoch: {}, loss: {}'.format(idx, loss_v[idx]))

    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))      
    return idx

def get_mean_index(y_prob, ev, col, shift):
    y = np.array(y_prob[:, col])
    return np.argmin(abs(y-(ev[col]+shift)/2))

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = '../../GN-data/dnn_train_sf.csv'
        valid_file = '../../GN-data/dnn_valid_sf.csv'
        test_file = '../../GN-data/dnn_test_sf.csv'
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
    num_dims = [20,12, 1]
    # num_dims = [34, 17, 1]
    npzFn = os.path.join(npzDir, "dnn_bce_20.npz")
    try:
        with np.load(npzFn) as f:
            PScr, RScr, FScr, nIter, AUC = f['PScr'], f['RScr'], f['FScr'], f['nIter'], f['AUC']
    except FileNotFoundError:
        ntest = 50
        patience = 20
        PScr = np.zeros(ntest)
        RScr = np.zeros(ntest)
        FScr = np.zeros(ntest)
        nIter = np.zeros(ntest)
        AUC = np.zeros(ntest)

        for i in range(ntest):
            print('Computing {}/{}...'.format(i+1, ntest))
            model = Stroke_DNN_Model(num_dims)
            loss_func = nn.BCELoss()
            opt = optim.Adam(model.parameters(), lr=lr,)
            early_stopping = EarlyStopping(patience, verbose=False)

            niter = fit(epochs, model, loss_func, opt, train_dl, valid_dl, valid_ds, early_stopping)
            print('Trained with {} iterations...'.format(niter))
            y_prob = model(valid_ds.data)
            y_prob = y_prob.squeeze().tolist()
            y_pred = np.array([0 if y_prob[i] < 0.5 else 1 for i in range(len(y_prob))])
            y_test = valid_ds.labels.tolist()
            PScr[i] = precision_score(y_test, y_pred, average='macro')
            RScr[i] = recall_score(y_test, y_pred, average='macro')
            FScr[i] = f1_score(y_test, y_pred, average='macro')
            nIter[i] = niter
            AUC[i] = roc_auc_score(y_test, y_prob)

        delIdx = np.where(abs(PScr) < 1e-5)[0]
        print('remove {} records...'.format(len(delIdx)))
        PScr = np.delete(PScr, delIdx, 0)
        RScr = np.delete(RScr, delIdx, 0)
        FScr = np.delete(FScr, delIdx, 0)
        nIter = np.delete(nIter, delIdx, 0)
        AUC = np.delete(AUC, delIdx, 0)
        np.savez(npzFn, PScr=PScr, RScr=RScr, FScr=FScr, nIter=nIter, AUC=AUC)

    mP, sP, PCIL, PCIR = statistics(PScr)
    mR, sR, RCIL, RCIR = statistics(RScr)
    mF, sF, FCIL, FCIR = statistics(FScr)
    mI, sI, ICIL, ICIR = statistics(nIter)
    mA, sA, ACIL, ACIR = statistics(AUC)
    
    tb = pt.PrettyTable()
    tb.field_names = ["DNN-BE", "Precision", "std", "CIL", "CIR"]
    tb.add_row(['Attack', mP, sP, PCIL, PCIR])
    print(tb)
    tb = pt.PrettyTable()
    tb.field_names = ["DNN-BE", "Recall", "std", "CIL", "CIR"]
    tb.add_row(['Attack', mR, sR, RCIL, RCIR])
    print(tb)
    tb = pt.PrettyTable()
    tb.field_names = ["DNN-BE", "F1-score", "std", "CIL", "CIR"]
    tb.add_row(['Attack', mF, sF, FCIL, FCIR])
    print(tb)
    tb = pt.PrettyTable()
    tb.field_names = ["DNN-BE", "AUC", "std", "CIL", "CIR"]
    tb.add_row(['Attack', mA, sA, ACIL, ACIR])
    print(tb)
    tb = pt.PrettyTable()
    tb.field_names = ["DNN-BE", "Iteration", "std", "CIL", "CIR"]
    tb.add_row(['Attack', mI, sI, ICIL, ICIR])
    print(tb)
