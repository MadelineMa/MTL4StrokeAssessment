# from shutil import get_archive_formats
import torch
import os
import numpy as np
import prettytable as pt
from torch import nn
from torch import optim
from sklearn.metrics import precision_score, recall_score, f1_score #, classification_report
from early_stopping import EarlyStopping
from utils import get_data, fit, statistics
from model import DNN

dnnDir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'posix':
   ptDir = os.path.join(dnnDir, "pt/")
   npzDir = os.path.join(dnnDir, "npz/")
else:
   ptDir = os.path.join(dnnDir, "pt\\")
   npzDir = op.path.join(dnnDir, "npz\\")

batch_size = 256

train_loader, valid_loader, test_x, test_y = get_data(batch_size)

# parameter set
lr = 1e-2
epochs = 30

num_dims = [784, 300, 100, 10]
model = DNN(num_dims, 10)

loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr,)

npzFn = os.path.join(npzDir, "dnn_ce.npz")
try:
    with np.load(npzFn) as f:
        PScr, RScr, FScr, nIter = f['PScr'], f['RScr'], f['FScr'], f['nIter']
except FileNotFoundError:
    ntest = 20
    patience = 20
    PScr = np.zeros((ntest,10))
    RScr = np.zeros((ntest,10))
    FScr = np.zeros((ntest,10))
    nIter = np.zeros(ntest)

    for i in range(ntest):
        print('Computing {}/{}...'.format(i+1, ntest))
        model = DNN(num_dims, 10)
        opt = optim.Adam(model.parameters(), lr=lr,)

        early_stopping = EarlyStopping(patience, verbose=False)

        niter = fit(epochs, model, loss_func, opt, train_loader, valid_loader, early_stopping)
        ptFn = os.path.join(ptDir, f"dnn_ce{i}.pt")
        torch.save(model.state_dict(), ptFn)
        print('Trained with {} iterations...'.format(niter))
        y_prob = model(test_x)
        y_prob = y_prob.detach().numpy()
        y_pred = np.argmax(y_prob.tolist(), axis=1)
        # report = classification_report(y_test, y_pred, digits=4)
        # print(report)
        PScr[i] = precision_score(test_y, y_pred, average=None)
        RScr[i] = recall_score(test_y, y_pred, average=None)
        FScr[i] = f1_score(test_y, y_pred, average=None)
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
ntable = np.array([0,1,2,3,4,5,6,7,8,9])
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
    tb.field_names = ["DNN-CE", "f1-score", "Fstd", "FCIL", "FCIR"]
    tb.add_row([ntable[i], mF[i], sF[i], FCIL[i], FCIR[i]])
print(tb)
tb = pt.PrettyTable()
tb.field_names = ["DNN-CE", "#iteration", "Istd", "ICIL", "ICIR"]
tb.add_row(['iteration', mI, sI, ICIL, ICIR])
print(tb)