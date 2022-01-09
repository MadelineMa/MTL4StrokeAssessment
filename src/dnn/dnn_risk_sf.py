import numpy as np
import sys
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
from strokeDataset import Dnn20
from sklearn.metrics import classification_report
from early_stopping import EarlyStopping
epsilon = torch.tensor(sys.float_info.epsilon)

class Stroke_DNN_Model(nn.Module):
    def __init__(self, num_dims):
        super().__init__()
        # num_dims = [20, 11, 4]
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
    plt.title('loss-dnn')
    plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))      

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
    train_ds = Dnn20(train_file)
    valid_ds = Dnn20(valid_file, train_ds.scaler)
    test_ds = Dnn20(test_file, train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    num_dims = [20, 11, 4]
    model = Stroke_DNN_Model(num_dims)
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