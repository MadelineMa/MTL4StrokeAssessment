from shutil import get_archive_formats
import torch
import os
import numpy as np
import prettytable as pt
from torch import nn
from torch import optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from early_stopping import EarlyStopping
from utils import get_data_binary, fit, statistics
from model import DNN

dnnDir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'posix':
   ptDir = os.path.join(dnnDir, "pt/")
   npzDir = os.path.join(dnnDir, "npz/")
else:
   ptDir = os.path.join(dnnDir, "pt\\")
   npzDir = op.path.join(dnnDir, "npz\\")

batch_size = 256

train_loader, valid_loader, test_x, test_y = get_data_binary(batch_size)

# parameter set
lr = 1e-2
epochs = 200

num_dims = [784, 300, 100, 1]
model = DNN(num_dims, 1)

loss_func = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=lr,)

npzFn = os.path.join(npzDir, "dnn_be.npz")
try:
   with np.load(npzFn) as f:
      PScr, RScr, FScr, nIter, AUC = f['PScr'], f['RScr'], f['FScr'], f['nIter'], f['AUC']
except FileNotFoundError:
   ntest = 20
   patience = 10
   PScr = np.zeros(ntest)
   RScr = np.zeros(ntest)
   FScr = np.zeros(ntest)
   nIter = np.zeros(ntest)
   AUC = np.zeros(ntest)

   for i in range(ntest):
      print('Computing {}/{}...'.format(i+1, ntest))
      model = DNN(num_dims, 1)
      opt = optim.Adam(model.parameters(), lr=lr,)

      early_stopping = EarlyStopping(patience, verbose=False)

      niter = fit(epochs, model, loss_func, opt, train_loader, valid_loader, early_stopping)
      ptFn = os.path.join(ptDir, f"dnn_be{i}.pt")
      torch.save(model.state_dict(), ptFn)
      print('Trained with {} iterations...'.format(niter))
      y_prob = model(test_x)
      y_prob = y_prob.detach().numpy()
      y_pred = [1 if y_prob[i] > 0.5 else 0 for i in range(len(y_prob))]
      # report = classification_report(y_test, y_pred, digits=4)
      # print(report)
      PScr[i] = precision_score(test_y, y_pred,  average='macro')
      RScr[i] = recall_score(test_y, y_pred,  average='macro')
      FScr[i] = f1_score(test_y, y_pred,  average='macro')
      nIter[i] = niter
      AUC[i] = roc_auc_score(test_y, y_prob)

   # PScr = np.delete(PScr, 0, 0) # MLiao: should delete "Precision is ill-defined and being set to 0.0 in labels" by hands...
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