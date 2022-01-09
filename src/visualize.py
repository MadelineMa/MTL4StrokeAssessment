import numpy as np
import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from strokeDataset import DnnDataset
from sklearn.metrics import classification_report
from early_stopping import EarlyStopping
from matplotlib import pyplot as plt
# epsilon = torch.tensor(sys.float_info.epsilon)

class Visulize_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # battom laryers
        self.lin1 = nn.Linear(34,17)
        # self.lin2 = nn.Linear(2, 4)
        # visulization layers
        self.linl1 = nn.Linear(17,2)
        self.linm1 = nn.Linear(17,2)
        self.linh1 = nn.Linear(17,2)
        self.lins1 = nn.Linear(17,2)
        # output layers
        self.linl2 = nn.Linear(2,1)
        self.linm2 = nn.Linear(2,1)
        self.linh2 = nn.Linear(2,1)
        self.lins2 = nn.Linear(2,1)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2)
        pl = self.linl1(x)
        pm = self.linm1(x)
        ph = self.linh1(x)
        ps = self.lins1(x)
        # output
        ol = F.relu(self.linl2(pl))
        om = F.relu(self.linm2(pm))
        oh = F.relu(self.linh2(ph))
        os = F.relu(self.lins2(ps))
        # x = torch.softmax(x, dim=1)
        return torch.cat([ol, om, oh, os], dim=1), pl, pm, ph, ps

def loss_batch(model, loss_func, x, y, opt=None):
    p_risk, _, _, _, _ = model(x)
    loss = loss_func(p_risk, y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, early_stopping=None):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        if epoch % 10 == 0:
            print(epoch, val_loss)
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))      
        

if __name__ == '__main__':
    if os.name == 'posix':
    #    train_file = '../GN-data/dnn_train.csv'
    #    valid_file = '../GN-data/dnn_valid.csv'
    #    test_file = '../GN-data/dnn_test.csv'
        train_file = './QI/data/qi_train.csv'
        valid_file = './QI/data/qi_valid.csv'
        test_file = './QI/data/qi_test.csv'
    else:
        train_file = '..\\GN-data\\dnn_train.csv'
        valid_file = '..\\GN-data\\dnn_valid.csv'
        test_file = '..\\GN-data\\dnn_test.csv'
    lr = 0.01
    bs = 100
    epochs = 400
    train_ds = DnnDataset(train_file)
    valid_ds = DnnDataset(valid_file, train_ds.scaler)
    test_ds = DnnDataset(test_file, train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)
    model = Visulize_Model()
    # loss_func = F.cross_entropy
    # loss_func = F.binary_cross_entropy
    loss_func = nn.CrossEntropyLoss()
    opt = optim.RMSprop(model.parameters(), lr=lr,)

    patience = 20	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)

    fit(epochs, model, loss_func, opt, train_dl, test_dl, early_stopping)
    y_prob, pl, pm, ph, ps = model(test_ds.data)
    # y_prob = y_prob.squeeze().tolist()
    # y_pred = np.array([0 if y_prob[i] < 0.5 else 1 for i in range(len(y_prob))])
    y_pred = np.argmax(y_prob.tolist(), axis=1)
    y_test = test_ds.labels.tolist()
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    psl = pl.detach().numpy()
    psm = pm.detach().numpy()
    psh = ph.detach().numpy()
    pss = ps.detach().numpy()

    y_test = np.array(y_test)

    psl = psl[np.where(y_test==0)[0], :]
    psm = psm[np.where(y_test==1)[0], :]
    psh = psh[np.where(y_test==2)[0], :]
    pss = pss[np.where(y_test==3)[0], :]

    plt.figure()
    p1 = plt.scatter(psl[:,0], psl[:,1], marker = 'o', color = 'blue', alpha = 0.5)
    p2 = plt.scatter(psm[:,0], psm[:,1], marker = 'x', color = 'green', alpha = 0.5)
    p3 = plt.scatter(psh[:,0], psh[:,1], marker = 'd', color = 'yellow', alpha = 0.5)
    p4 = plt.scatter(pss[:,0], pss[:,1], marker = '^', color = 'red', alpha = 0.5)
    plt.legend([p1, p2, p3, p4], ['low', 'medium', 'high', 'attack'], loc='best')
    plt.show()
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.xlim((-25,25))
    plt.ylim((-10,10))
    plt.scatter(psl[:,0], psl[:,1], marker = 'o', color = 'blue', alpha = 0.5)
    plt.subplot(2,2,2)
    plt.xlim((-25,25))
    plt.ylim((-10,10))
    plt.scatter(psm[:,0], psm[:,1], marker = 'x', color = 'green', alpha = 0.5)
    plt.subplot(2,2,3)
    plt.xlim((-25,25))
    plt.ylim((-10,10))
    plt.scatter(psh[:,0], psh[:,1], marker = 'd', color = 'yellow', alpha = 0.5)
    plt.subplot(2,2,4)
    plt.xlim((-25,25))
    plt.ylim((-10,10))
    plt.scatter(pss[:,0], pss[:,1], marker = '^', color = 'red', alpha = 0.5)
    plt.show()
