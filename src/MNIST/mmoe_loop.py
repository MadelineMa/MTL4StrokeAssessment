import torch
import os
import numpy as np
import prettytable as pt
from torch import nn
from torch import optim
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from numpy.random import choice, seed
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from early_stopping import EarlyStopping
from model import MMOE

dnnDir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'posix':
   ptDir = os.path.join(dnnDir, "pt/")
   npzDir = os.path.join(dnnDir, "npz/")
else:
   ptDir = os.path.join(dnnDir, "pt\\")
   npzDir = op.path.join(dnnDir, "npz\\")

def statistics(X, alpha=0.95):
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    SE = sig/np.sqrt(X.shape[0])
    CI = stats.norm.interval(alpha, loc=mu, scale=SE)
    return mu, sig, CI[0], CI[1]

def get_data_mmoe(batch_size, k=0.1):
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)
    X = train_dataset.data
    y = train_dataset.targets

    k = int(y.shape[0] * k)
    seed(42)
    idx = choice(range(X.shape[0]), size=k, replace=False)
    X = X[idx,:,:]
    yb = torch.tensor([1 if y[i]==0 else 0 for i in idx])
    y = y[idx]
    k = 3000
    Xt = test_dataset.data
    yt = test_dataset.targets
    seed(42)
    vidx = choice(range(Xt.shape[0]), size=k, replace=False)
    Xv = Xt[vidx,:,:]
    tidx = range(Xt.shape[0])
    tidx = np.setdiff1d(tidx, vidx)
    Xt = Xt[tidx,:,:]
    yv = yt[vidx]
    yt = yt[tidx]
    ytb = torch.tensor([1 if yt[i]==0 else 0 for i in range(yt.shape[0])])
    yvb = torch.tensor([1 if yv[i]==0 else 0 for i in range(yv.shape[0])])

    train_x, train_y, train_yb = [
        X.view(X.shape[0],-1).float(),
        y.long(),
        yb.float().unsqueeze_(-1)
    ]

    valid_x, valid_y, valid_yb = [
        Xv.view(Xv.shape[0],-1).float(),
        yv.long(),
        yvb.float().unsqueeze_(-1)
    ]
    test_x, test_y, test_yb = [
        Xt.view(Xt.shape[0],-1).float(),
        yt.long(),
        ytb.float().unsqueeze_(-1)
    ]
    train_dataset = TensorDataset(train_x, train_y, train_yb)
    valid_dataset = TensorDataset(valid_x, valid_y, valid_yb)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=batch_size)
    return train_loader, valid_loader, test_x, test_y, test_yb

def loss_batch(model, loss_qi_fun, loss_dnn_fun, xv, y, yb, opt=None):
    p_bce, p_ce = model(xv)
    loss_bce = loss_dnn_fun(p_bce, yb)
    loss_ce = loss_qi_fun(p_ce, y)

    loss = 1*loss_bce + 1*loss_ce

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), loss_bce.item(), loss_ce.item(), len(xv)

def fit(epochs, model, loss_qi_fun, loss_dnn_fun, opt, train_dl, valid_dl, early_stopping=None):
    loss_v = []
    loss_bce_v = []
    loss_ce_v = []
    auc_v = []
    for epoch in range(epochs):
        model.train()
        for xvb, y, yb in train_dl:
            loss_batch(model, loss_qi_fun, loss_dnn_fun, xvb, y, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, losses_bce, losses_ce, nums = zip(*[loss_batch(model, loss_qi_fun, loss_dnn_fun, xvb, y, yb) for xvb, y, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_loss_bce = np.sum(np.multiply(losses_bce, nums)) / np.sum(nums)
        val_loss_ce = np.sum(np.multiply(losses_ce, nums)) / np.sum(nums)
        loss_v.append(val_loss)
        loss_bce_v.append(val_loss_bce)
        loss_ce_v.append(val_loss_ce)

        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    lidx = np.argmin(loss_v)
    lbidx = np.argmin(loss_bce_v)
    lcidx = np.argmin(loss_ce_v)
    # aucidx = np.argmin(auc_v)
    print('minimum: ')
    print(f'epoch: {lidx}, loss: {loss_v[lidx]}')
    print(f'epoch: {lbidx}, loss_bce: {loss_bce_v[lbidx]}')
    print(f'epoch: {lcidx}, loss_ce: {loss_ce_v[lcidx]}')
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt')) 
    return lidx

batch_size = 256

train_loader, valid_loader, test_x, test_y, test_yb = get_data_mmoe(batch_size)

# parameter set
lr = 1e-2
epochs = 400
feature_idx = [433, 349, 350]
qi_dims = [784, 300, 100, 10]
dnn_dims = [784, 300, 100, 1]
model = MMOE(feature_idx, qi_dims, dnn_dims)

loss_dnn_func = nn.BCELoss()
loss_qi_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr)
npzFn = os.path.join(npzDir, "mmoe.npz")
try:
    with np.load(npzFn) as f:
        PScr, RScr, FScr, nIter, AUC, PScrB, RScrB, FScrB = f['PScr'], f['RScr'], f['FScr'], f['nIter'], f['AUC'], f['PScrB'], f['RScrB'], f['FScrB']
except FileNotFoundError:
    ntest = 20
    patience = 20
    PScr = np.zeros((ntest,10))
    RScr = np.zeros((ntest,10))
    FScr = np.zeros((ntest,10))
    PScrB = np.zeros(ntest)
    RScrB = np.zeros(ntest)
    FScrB = np.zeros(ntest)
    nIter = np.zeros(ntest)
    AUC = np.zeros(ntest)

    for i in range(ntest):
        print('Computing {}/{}...'.format(i+1, ntest))
        model = MMOE(feature_idx, qi_dims, dnn_dims)
        opt = optim.Adam(model.parameters(), lr=lr,)

        early_stopping = EarlyStopping(patience, verbose=False)

        niter = fit(epochs, model, loss_qi_func, loss_dnn_func, opt, train_loader, valid_loader, early_stopping)
        ptFn = os.path.join(ptDir, f"mmoe{i}.pt")
        torch.save(model.state_dict(), ptFn)
        print('Trained with {} iterations...'.format(niter))
        yb_prob, y_prob = model(test_x)
        y_prob = y_prob.detach().numpy()
        yb_prob = yb_prob.detach().numpy()
        yb_pred = [1 if yb_prob[i] > 0.5 else 0 for i in range(len(yb_prob))]
        y_pred = np.argmax(y_prob.tolist(), axis=1)
        # report = classification_report(y_test, y_pred, digits=4)
        # print(report)
        PScr[i] = precision_score(test_y, y_pred, average=None)
        RScr[i] = recall_score(test_y, y_pred, average=None)
        FScr[i] = f1_score(test_y, y_pred, average=None)
        PScrB[i] = precision_score(test_yb, yb_pred,  average='macro')
        RScrB[i] = recall_score(test_yb, yb_pred,  average='macro')
        FScrB[i] = f1_score(test_yb, yb_pred,  average='macro')
        nIter[i] = niter
        AUC[i] = roc_auc_score(test_yb, yb_prob)

    # PScr = np.delete(PScr, 0, 0) # MLiao: should delete "Precision is ill-defined and being set to 0.0 in labels" by hands...
    delIdx = np.where(abs(PScr) < 1e-5)[0]
    print('remove {} records...'.format(len(delIdx)))
    PScr = np.delete(PScr, delIdx, 0)
    RScr = np.delete(RScr, delIdx, 0)
    FScr = np.delete(FScr, delIdx, 0)
    PScrB = np.delete(PScrB, delIdx, 0)
    RScrB = np.delete(RScrB, delIdx, 0)
    FScrB = np.delete(FScrB, delIdx, 0)
    nIter = np.delete(nIter, delIdx, 0)
    AUC = np.delete(AUC, delIdx, 0)
    np.savez(npzFn, PScr=PScr, RScr=RScr, FScr=FScr, nIter=nIter, PScrB=PScrB, RScrB=RScrB, FScrB=FScrB, AUC=AUC)

mPb, sPb, PCILb, PCIRb = statistics(PScrB)
mRb, sRb, RCILb, RCIRb = statistics(RScrB)
mFb, sFb, FCILb, FCIRb = statistics(FScrB)
mI, sI, ICIL, ICIR = statistics(nIter)
mA, sA, ACIL, ACIR = statistics(AUC)

tb = pt.PrettyTable()
tb.field_names = ["MMOE", "Precision", "std", "CIL", "CIR"]
tb.add_row(['Attack', mPb, sPb, PCILb, PCIRb])
print(tb)
tb = pt.PrettyTable()
tb.field_names = ["MMOE", "Recall", "std", "CIL", "CIR"]
tb.add_row(['Attack', mRb, sRb, RCILb, RCIRb])
print(tb)
tb = pt.PrettyTable()
tb.field_names = ["MMOE", "F1-score", "std", "CIL", "CIR"]
tb.add_row(['Attack', mFb, sFb, FCILb, FCIRb])
print(tb)
tb = pt.PrettyTable()
tb.field_names = ["MMOE", "AUC", "std", "CIL", "CIR"]
tb.add_row(['Attack', mA, sA, ACIL, ACIR])
print(tb)
tb = pt.PrettyTable()
tb.field_names = ["MMOE", "#iteration", "Istd", "ICIL", "ICIR"]
tb.add_row(['iteration', mI, sI, ICIL, ICIR])
print(tb)

mP, sP, PCIL, PCIR = statistics(PScr)
mR, sR, RCIL, RCIR = statistics(RScr)
mF, sF, FCIL, FCIR = statistics(FScr)

tb = pt.PrettyTable()
ntable = np.array([0,1,2,3,4,5,6,7,8,9])
for i in range(len(ntable)):
    tb.field_names = ["MMOE", "precision", "std", "CIL", "CIR"]
    tb.add_row([ntable[i], mP[i], sP[i], PCIL[i], PCIR[i]])
print(tb)
tb = pt.PrettyTable()
for i in range(len(ntable)):
    tb.field_names = ["MMOe", "Recall", "std", "CIL", "CIR"]
    tb.add_row([ntable[i], mR[i], sR[i], RCIL[i], RCIR[i]])
print(tb)
tb = pt.PrettyTable()
for i in range(len(ntable)):
    tb.field_names = ["MMOE", "F1-score", "std", "CIL", "CIR"]
    tb.add_row([ntable[i], mF[i], sF[i], FCIL[i], FCIR[i]])
print(tb)