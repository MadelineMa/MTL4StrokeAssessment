import numpy as np
import sys
import torch
import shap
from torch import tensor
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from scipy.special import comb
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
        ## QI part
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )
        # dnn part
        for i in range(1, 2): # consider just one hidden layer due to lack of samples
            setattr(self, 'lin'+str(i), 
                    nn.Linear(mpl_dims[i-1], mpl_dims[i]))
            setattr(self, 'relu'+str(i),
                    nn.ReLU())
            setattr(self, 'dropout'+str(i),
                    nn.Dropout(p=drop_rate))
        # self.softmax = nn.Softmax(dim=1)
        self.lin2 = nn.Linear(mpl_dims[1] + int(comb(self.field_size, 2)), mpl_dims[-1])

    def forward(self, xv, xi):
        qi_emb_x = []
        for i, emb in enumerate(self.embeddings):
            qi_emb_x.append(emb(xi[:,i]) * xv[:,i].unsqueeze(1)) 
        qi_sum_emb_x = sum(qi_emb_x)
        emb_sum_square = qi_sum_emb_x * qi_sum_emb_x
        emb_x_square = [item * item for item in qi_emb_x]
        emb_square_sum = sum(emb_x_square)
        qi = (emb_sum_square - emb_square_sum) * 0.5

        for i in range(1, 2):
            x = getattr(self, 'lin'+str(i))(xv)
            x = getattr(self, 'relu'+str(i))(x)
            x = getattr(self, 'dropout'+str(i))(x)
        x = torch.cat([qi, x])
        x = self.lin2(x)
        return x

def loss_batch(model, loss_func, xv, xi, y, opt=None):
    p_risk = model(xv, xi)
    loss = loss_func(p_risk, y)  

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xv)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, early_stopping=None):
    for epoch in range(epochs):
        model.train()
        for xvb, xib, yb in train_dl:
            loss_batch(model, loss_func, xvb, xib, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xvb, xib, yb) for xvb, xib, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))              

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = '../../GN-data/dnn_train_sf.csv'
        valid_file = '../../GN-data/dnn_valid_sf.csv'
        test_file = '../../GN-data/dnn_test_sf.csv'
        train_file_ind = '../../GN-data/dnn_train_sf_ind.csv'
        valid_file_ind = '../../GN-data/dnn_valid_sf_ind.csv'
        test_file_ind = '../../GN-data/dnn_test_sf_ind.csv'
    else:
        train_file = '..\\..\\GN-data\\dnn_train_sf.csv'
        valid_file = '..\\..\\GN-data\\dnn_valid_sf.csv'
        test_file = '..\\..\\GN-data\\dnn_test_sf.csv'
        train_file_ind = '..\\..\\GN-data\\dnn_train_sf_ind.csv'
        valid_file_ind = '..\\..\\GN-data\\dnn_valid_sf_ind.csv'
        test_file_ind = '..\\..\\GN-data\\dnn_test_sf_ind.csv'
    lr = 0.01
    bs = 3
    epochs = 400
    train_ds = QIDataset(train_file, train_file_ind)
    valid_ds = QIDataset(valid_file, valid_file_ind, train_ds.scaler)
    test_ds = QIDataset(test_file, test_file_ind ,train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    features = ['LSBP', 'Exs', 'Sm']           # to decide
    feature_sizes = [146, 2, 3]      # to compute according to features
    cols = train_ds.dataframe.columns.tolist()
    feature_index = [cols.index(features[0])]
    for i in range(1,len(features)):
        feature_index.append(cols.index(features[i]))
    mpl_dims = [20, 16, 4]  # check this
    model = DeepQI(feature_sizes, mpl_dims)
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr,)

    patience = 20	
    early_stopping = EarlyStopping(patience, verbose=False)

    fit(epochs, model, loss_func, opt, train_dl, test_dl, early_stopping)
    y_prob = model(test_ds.data)
    y_pred = np.argmax(y_prob.tolist(), axis=1)
    y_test = test_ds.labels.tolist()
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
