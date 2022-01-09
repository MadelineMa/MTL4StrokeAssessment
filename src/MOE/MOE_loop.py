import numpy as np
import os
import sys
import torch
import copy
import prettytable as pt

from torch import tensor
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

if os.name == 'posix':
    sys.path.append("../")
else:
    sys.path.append("..\\")
from strokeDataset import MoEDataset
from early_stopping import EarlyStopping
from utils import statistics

dnnDir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'posix':
   ptDir = os.path.join(dnnDir, "pt/")
   npzDir = os.path.join(dnnDir, "npz/")
else:
   ptDir = os.path.join(dnnDir, "pt\\")
   npzDir = op.path.join(dnnDir, "npz\\")

class MOE(nn.Module):
    def __init__(self, feature_sizes, qi_dims, dnn_dims, embedding_size=4, drop_rate=0.2):
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.qi_dims = qi_dims
        self.dnn_dims = dnn_dims
        comb_fea_index = []
        for idx in combinations(range(self.field_size), 2):
            comb_fea_index.append(list(idx))
        ## QI part
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )
        self.comb_fea_index = comb_fea_index
        # QI-dnn part
        self.QIlin1 = nn.Linear(qi_dims[0] + len(comb_fea_index), qi_dims[1])
        self.QIrelu1 = nn.ReLU()
        self.QIdrop1 = nn.Dropout(p=drop_rate)
        ## leave here for further extension
        # for i in range(2, 3): # consider just one hidden layer due to lack of samples
        #     setattr(self, 'QIlin'+str(i), 
        #             nn.Linear(qi_dims[i-1], qi_dims[i]))
        #     setattr(self, 'QIrelu'+str(i),
        #             nn.ReLU())
        #     setattr(self, 'QIdrop'+str(i),
        #             nn.Dropout(p=drop_rate))
        self.QIout = nn.Linear(qi_dims[-2], qi_dims[-1])
        # pure dnn part
        for i in range(1, len(dnn_dims)-1):
            setattr(self, 'DNNlin'+str(i),
                    nn.Linear(dnn_dims[i-1], dnn_dims[i]))
            setattr(self, 'DNNrelu'+str(i),
                    nn.ReLU())
            setattr(self, 'DNNdrop'+str(i),
                    nn.Dropout(p=drop_rate))
        self.DNNOut = nn.Linear(dnn_dims[-2], dnn_dims[-1])
        # gate 34 -> 2 ?? TODO: need more layers?
        for i in range(1, 3): # number of towers, fixed to 2.
            setattr(self, 'GATE'+str(i)+'lin',
                    nn.Linear(dnn_dims[0], 2))
            setattr(self, 'GATE'+str(i)+'softmax',
                    nn.Softmax(dim=1))   

    def forward(self, xv, xi):
        # QI part
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(xi[:,i]))
        embs = torch.stack(embs)
        qi = []
        for pair in self.comb_fea_index:
            i, j = pair
            qi.append((embs[i] * embs[j]).sum(1).unsqueeze(1))
        qi = torch.cat(qi, dim=1)

        xqi = torch.cat([qi, xv], dim=1)
        for i in range(1, 2):
            xqi = getattr(self, 'QIlin'+str(i))(xqi)
            xqi = getattr(self, 'QIrelu'+str(i))(xqi)
            xqi = getattr(self, 'QIdrop'+str(i))(xqi)
        # xqi = self.QIout(xqi)

        # pure dnn part
        xdnn = copy.deepcopy(xv)
        for i in range(1, len(self.dnn_dims)-1):
            xdnn = getattr(self, 'DNNlin'+str(i))(xdnn)
            xdnn = getattr(self, 'DNNrelu'+str(i))(xdnn)
            xdnn = getattr(self, 'DNNdrop'+str(i))(xdnn)
        # gate
        x = torch.stack([torch.rand(xdnn.shape), torch.rand(xqi.shape)])
        for i in range(1, 3):
            g = getattr(self, 'GATE'+str(i)+'lin')(xv)
            g = getattr(self, 'GATE'+str(i)+'softmax')(g)
            x[i-1] = g[:,0].unsqueeze(dim=1)*xqi + g[:,1].unsqueeze(dim=1)*xdnn
            # xdnn = getattr(self, 'DNNdrop'+str(i))(xdnn)
        # gate offers weights to connect QI and DNN
        # TODO: check if x[0] == x[1] to see how g will be constructed.
        xqi = self.QIout(x[0])
        i = len(self.dnn_dims)-2
        xdnn = self.DNNOut(x[1])
        return torch.sigmoid(xdnn), xqi

def loss_batch(model, loss_qi_fun, loss_dnn_fun, xv, xi, ys, yr, opt=None):
    p_bce, p_ce = model(xv, xi)
    loss_bce = loss_dnn_fun(p_bce, ys)
    loss_ce = loss_qi_fun(p_ce, yr)

    loss = 1*loss_bce + 1*loss_ce

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), loss_bce.item(), loss_ce.item(), len(xv)

def fit(epochs, model, loss_qi_fun, loss_dnn_fun, opt, train_dl, valid_dl, valid_ds, feature_index, early_stopping=None):
    loss_v = []
    loss_bce_v = []
    loss_ce_v = []
    auc_v = []
    y_slabel = np.array(valid_ds.slabels)
    x_valid_data = valid_ds.data
    x_valid_index = valid_ds.index[:, feature_index]
    for epoch in range(epochs):
        model.train()
        for xvb, xib, ysb, yrb in train_dl:
            xib = xib[:, feature_index]
            loss_batch(model, loss_qi_fun, loss_dnn_fun, xvb, xib, ysb, yrb, opt)

        model.eval()
        with torch.no_grad():
            losses, losses_bce, losses_ce, nums = zip(*[loss_batch(model, loss_qi_fun, loss_dnn_fun, xvb, xib[:,feature_index], ysb, yrb) for xvb, xib, ysb, yrb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_loss_bce = np.sum(np.multiply(losses_bce, nums)) / np.sum(nums)
        val_loss_ce = np.sum(np.multiply(losses_ce, nums)) / np.sum(nums)
        loss_v.append(val_loss)
        loss_bce_v.append(val_loss_bce)
        loss_ce_v.append(val_loss_ce)
        # print(epoch, val_loss)
        y_s_prob, _ = model(x_valid_data, x_valid_index)
        auc_v.append(roc_auc_score(y_slabel, y_s_prob.detach().numpy()))
        # dnn_fpr, dnn_tpr, thresholds = roc_curve(y_slabel, y_s_prob.detach().numpy(), pos_label=1)
        # dnn_auc = auc(dnn_fpr, dnn_tpr)
        # auc_v.append(dnn_auc)

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
    print(f'epoch: {lidx}, auc: {auc_v[lidx]}')
    # _, axes = plt.subplots(2,2, figsize=(6, 6))
    # colors = ['y', 'b', 'y', 'y']

    # ylabels = ['loss', 'auc of expert 1', 'loss of expert 1', 'loss of expert 2']
    # # indices = [lidx, lidx, lbidx, lcidx]
    # indices = [lidx, lidx, lidx, lidx]
    # plt_data = [np.array(loss_v), np.array(auc_v), np.array(loss_bce_v), np.array(loss_ce_v)]
    # # for ax, c, t, d, idx in zip(axes.flatten(), colors, title, plt_data, indices):
    # #     ax.plot(np.arange(len(d)), d, c=c)
    # #     ax.plot(idx, d[idx], marker='o')
    # #     ax.set_title(t)
    # for d, idx, ylabel in zip(plt_data, indices, ylabels):
    #     plt.figure()
    #     plt.plot(np.arange(len(d)), d, c='b')
    #     plt.plot(idx, d[idx], marker='o', c='r')
    #     plt.xlabel('epoches')
    #     plt.ylabel(ylabel)
    #     plt.show()

    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))      
    return lidx        

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = './data/train_en.csv'
        valid_file = './data/valid_en.csv'
        test_file = './data/test_en.csv'
        train_file_ind = './data/train_en_ind.csv'
        valid_file_ind = './data/valid_en_ind.csv'
        test_file_ind = './data/test_en_ind.csv'
    else:
        train_file = '.\\data\\train_en.csv'
        valid_file = '.\\data\\valid_en.csv'
        test_file = '.\\data\\test_en.csv'
        train_file_ind = '.\\data\\train_en_ind.csv'
        valid_file_ind = '.\\data\\valid_en_ind.csv'
        test_file_ind = '.\\data\\test_en_ind.csv'
    lr = 0.01
    bs = 100
    epochs = 400
    train_ds = MoEDataset(train_file, train_file_ind)
    valid_ds = MoEDataset(valid_file, valid_file_ind, train_ds.scaler)
    test_ds = MoEDataset(test_file, test_file_ind ,train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    features = ['LSBP', 'Exs', 'Sm']  
    # feature_sizes = [146, 2, 3] 
    features_sizes = np.load('./data/feature_sizes.npy')
    cols = train_ds.dataframe.columns.tolist()
    feature_index = [cols.index(features[0])]
    for i in range(1,len(features)):
        feature_index.append(cols.index(features[i]))
    qi_dims = [20, 11, 4]  # check this
    dnn_dims = [20, 11, 1]
    feature_sizes = features_sizes[feature_index]

    npzFn = os.path.join(npzDir, "mmoe.npz")
    try:
        with np.load(npzFn) as f:
            PScr1, RScr1, FScr1, nIter, AUC, PScr2, RScr2, FScr2 = f['PScr1'], f['RScr1'], f['FScr1'], f['nIter'], f['AUC'], f['PScr2'], f['RScr2'], f['FScr2']
    except FileNotFoundError:
        ntest = 50
        patience = 30
        PScr1 = np.zeros(ntest)
        RScr1 = np.zeros(ntest)
        FScr1 = np.zeros(ntest)
        AUC = np.zeros(ntest)
        nIter = np.zeros(ntest)

        PScr2 = np.zeros((ntest,4))
        RScr2 = np.zeros((ntest,4))
        FScr2 = np.zeros((ntest,4))

        for i in range(ntest):
            print('Computing {}/{}...'.format(i+1, ntest))
            model = MOE(feature_sizes, qi_dims, dnn_dims)

            loss_dnn_fun = nn.BCELoss()
            loss_qi_fun = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=lr,)
            early_stopping = EarlyStopping(patience, verbose=False)

            niter = fit(epochs, model, loss_qi_fun, loss_dnn_fun, opt, train_dl, valid_dl, valid_ds, feature_index, early_stopping)
            y_s_prob, y_r_prob = model(test_ds.data, test_ds.index[:,feature_index])
            y_s_prob = y_s_prob.squeeze().tolist()
            y_s_pred = np.array([0 if y_s_prob[i] < 0.5 else 1 for i in range(len(y_s_prob))])
            y_r_pred = np.argmax(y_r_prob.tolist(), axis=1)
            y_r_test = test_ds.rlabels.tolist()
            y_s_test = test_ds.slabels.tolist()
            PScr1[i] = precision_score(y_s_test, y_s_pred, average='macro')
            RScr1[i] = recall_score(y_s_test, y_s_pred, average='macro')
            FScr1[i] = f1_score(y_s_test, y_s_pred, average='macro')
            nIter[i] = niter
            AUC[i] = roc_auc_score(y_s_test, y_s_prob)
            PScr2[i] = precision_score(y_r_test, y_r_pred, average=None)
            RScr2[i] = recall_score(y_r_test, y_r_pred, average=None)
            FScr2[i] = f1_score(y_r_test, y_r_pred, average=None)

        delIdx = np.where(abs(PScr1) < 1e-5)[0]
        print('remove {} records...'.format(len(delIdx)))
        PScr1 = np.delete(PScr1, delIdx, 0)
        RScr1 = np.delete(RScr1, delIdx, 0)
        FScr1 = np.delete(FScr1, delIdx, 0)
        nIter = np.delete(nIter, delIdx, 0)
        AUC = np.delete(AUC, delIdx, 0)
        PScr2 = np.delete(PScr2, delIdx, 0)
        RScr2 = np.delete(RScr2, delIdx, 0)
        FScr2 = np.delete(FScr2, delIdx, 0)

        mPScr2 = PScr2.mean(axis=0)
        mRScr2 = RScr2.mean(axis=0)
        mFScr2 = FScr2.mean(axis=0)

        np.savez(npzFn, PScr1=PScr1, RScr1=RScr1, FScr1=FScr1, nIter=nIter, AUC=AUC, PScr2=PScr2, RScr2=RScr2, FScr2=FScr2)

    mPb, sPb, PCILb, PCIRb = statistics(PScr1)
    mRb, sRb, RCILb, RCIRb = statistics(RScr1)
    mFb, sFb, FCILb, FCIRb = statistics(FScr1)
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

    mP, sP, PCIL, PCIR = statistics(PScr2)
    mR, sR, RCIL, RCIR = statistics(RScr2)
    mF, sF, FCIL, FCIR = statistics(FScr2)

    tb = pt.PrettyTable()
    ntable = ['Low', 'Medium', 'High', 'Attact']
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