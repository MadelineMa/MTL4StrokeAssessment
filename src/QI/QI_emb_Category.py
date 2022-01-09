import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy.core.fromnumeric import argmax
import torch
import shap
from torch import tensor
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from itertools import combinations
import os
if os.name == 'posix':
    sys.path.append("../")
else:
    sys.path.append("..\\")
from strokeDataset import QIDataset
from sklearn.metrics import classification_report
from early_stopping import EarlyStopping
epsilon = torch.tensor(sys.float_info.epsilon)

class DeepQI(nn.Module):
    def __init__(self, feature_sizes, mpl_dims, embedding_size=4, drop_rate=0.2):
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.mpl_dims = mpl_dims
        comb_fea_index = []
        for idx in combinations(range(self.field_size), 2):
            comb_fea_index.append(list(idx))
        ## QI part
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )
        self.comb_fea_index = comb_fea_index
        # dnn part
        self.lin1 = nn.Linear(mpl_dims[0] + len(comb_fea_index), mpl_dims[1])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=drop_rate)
        # for i in range(2, 3): # consider just one hidden layer due to lack of samples
        #     setattr(self, 'lin'+str(i), 
        #             nn.Linear(mpl_dims[i-1], mpl_dims[i]))
        #     setattr(self, 'relu'+str(i),
        #             nn.ReLU())
        #     setattr(self, 'dropout'+str(i),
        #             nn.Dropout(p=drop_rate))
        # self.softmax = nn.Softmax(dim=1)
        # self.lin2 = nn.Linear(mpl_dims[1] + int(comb(self.field_size, 2)), mpl_dims[-1])
        # self.lin2 = nn.Linear(mpl_dims[1] + len(comb_fea_index), mpl_dims[-1])
        self.output = nn.Linear(mpl_dims[-2], mpl_dims[-1])

    def forward(self, xv, xi):
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(xi[:,i]))
        embs = torch.stack(embs)
        qi = []
        for pair in self.comb_fea_index:
            i, j = pair
            qi.append((embs[i] * embs[j]).sum(1).unsqueeze(1))
        qi = torch.cat(qi, dim=1)

        xv = torch.cat([qi, xv], dim=1)
        for i in range(1, 2):
            xv = getattr(self, 'lin'+str(i))(xv)
            xv = getattr(self, 'relu'+str(i))(xv)
            xv = getattr(self, 'dropout'+str(i))(xv)
        # xv = self.lin2(xv)
        xv = self.output(xv)
        return xv

def loss_batch(model, loss_func, xv, xi, y, opt=None):
    p_risk = model(xv, xi)
    loss = loss_func(p_risk, y) 

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xv)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, feature_index, early_stopping=None):
    loss_v = []
    for epoch in range(epochs):
        model.train()
        for xvb, xib, yb in train_dl:
            xib = xib[:, feature_index]
            loss_batch(model, loss_func, xvb, xib, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xvb, xib[:,feature_index], yb) for xvb, xib, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        loss_v.append(val_loss)
        
        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    idx = np.argmin(loss_v)
    print('minimum: ')
    print(f'epoch: {idx}, loss: {loss_v[idx]}')
    plt.figure()
    plt.plot(np.arange(len(loss_v)), np.array(loss_v),c='y')
    plt.plot(idx, loss_v[idx], marker='o')
    plt.title('loss')
    plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))              

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = './data/qi_train.csv'
        valid_file = './data/qi_valid.csv'
        test_file = './data/qi_test.csv'
        train_file_ind = './data/qi_train_ind.csv'
        valid_file_ind = './data/qi_valid_ind.csv'
        test_file_ind = './data/qi_test_ind.csv'
    else:
        train_file = '.\\data\\qi_train.csv'
        valid_file = '.\\data\\qi_valid.csv'
        test_file = '.\\data\\qi_test.csv'
        train_file_ind = '.\\data\\qi_train_ind.csv'
        valid_file_ind = '.\\data\\qi_valid_ind.csv'
        test_file_ind = '.\\data\\qi_test_ind.csv'
    lr = 0.01
    bs = 100
    epochs = 400
    train_ds = QIDataset(train_file, train_file_ind)
    valid_ds = QIDataset(valid_file, valid_file_ind, train_ds.scaler)
    test_ds = QIDataset(test_file, test_file_ind ,train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    features = ['LSBP', 'Exs', 'Sm']#, 'LDBP']#, 'RSBP']#, 'HbA1c', 'HA']  
    features_sizes = np.load('./data/feature_sizes.npy') 
    cols = train_ds.dataframe.columns.tolist()
    feature_index = [cols.index(features[0])]
    for i in range(1,len(features)):
        feature_index.append(cols.index(features[i]))
    feature_sizes = features_sizes[feature_index]
    feature_sizes[np.where((feature_sizes<100) & (feature_sizes>7))] = 30  # MLiao: artificial! 30 from n_category in QI_data.py Edu has 7 classes
    mpl_dims = [34, 17, 4]
    model = DeepQI(feature_sizes, mpl_dims)
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr,)
    
    patience = 40	
    early_stopping = EarlyStopping(patience, verbose=False)

    fit(epochs, model, loss_func, opt, train_dl, valid_dl, feature_index, early_stopping)
    y_prob = model(test_ds.data, test_ds.index[:,feature_index])
    y_pred = np.argmax(y_prob.tolist(), axis=1)
    y_test = test_ds.labels.tolist()
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
