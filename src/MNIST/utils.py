from os import replace
import numpy as np
from numpy.random import choice, seed
import matplotlib.pyplot as plt
import torch
from torch.functional import Tensor
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

def get_data(batch_size, k=0.1):
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)
    X = train_dataset.data
    y = train_dataset.targets

    k = int(y.shape[0] * k)
    seed(42)
    idx = choice(range(X.shape[0]), size=k, replace=False)
    X = X[idx,:,:]
    y = y[idx]
    k = 3000
    seed(42)
    Xt = test_dataset.data
    yt = test_dataset.targets
    vidx = choice(range(Xt.shape[0]), size=k, replace=False)
    Xv = Xt[vidx,:,:]
    tidx = range(Xt.shape[0])
    tidx = np.setdiff1d(tidx, vidx)
    Xt = Xt[tidx,:,:]
    yv = yt[vidx]
    yt = yt[tidx]

    train_x, train_y = [
        X.view(X.shape[0],-1).float(),
        y.long()
    ]
    valid_x, valid_y = [
        Xv.view(Xv.shape[0],-1).float(),
        yv.long()
    ]
    test_x, test_y = [
        Xt.view(Xt.shape[0],-1).float(),
        yt.long()
    ]
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=batch_size)
    return train_loader, valid_loader, test_x, test_y

def get_data_binary(batch_size, k=0.1):
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)
    X = train_dataset.data
    y = train_dataset.targets

    k = int(y.shape[0] * k)
    seed(42)
    idx = choice(range(X.shape[0]), size=k, replace=False)
    X = X[idx,:,:]
    y = torch.tensor([1 if y[i]==0 else 0 for i in idx])
    k = 3000
    Xt = test_dataset.data
    yt = test_dataset.targets
    yt = torch.tensor([1 if yt[i]==0 else 0 for i in range(yt.shape[0])])
    seed(42)
    vidx = choice(range(Xt.shape[0]), size=k, replace=False)
    Xv = Xt[vidx,:,:]
    tidx = range(Xt.shape[0])
    tidx = np.setdiff1d(tidx, vidx)
    Xt = Xt[tidx,:,:]
    yv = yt[vidx]
    yt = yt[tidx]

    train_x, train_y = [
        X.view(X.shape[0],-1).float(),
        y.float().unsqueeze_(-1)
    ]

    valid_x, valid_y = [
        Xv.view(Xv.shape[0],-1).float(),
        yv.float().unsqueeze_(-1)
    ]
    test_x, test_y = [
        Xt.view(Xt.shape[0],-1).float(),
        yt.float().unsqueeze_(-1)
    ]
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=batch_size)
    return train_loader, valid_loader, test_x, test_y

def get_data_shap(k=1000):
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)
    X = train_dataset.data
    y = train_dataset.targets

    seed(42)
    idx = choice(range(X.shape[0]), size=k, replace=False)
    X = X[idx,:,:]
    y = y[idx]
    k = 3000
    seed(42)
    Xt = test_dataset.data
    yt = test_dataset.targets
    tidx = choice(range(Xt.shape[0]), size=k, replace=False)
    Xt = Xt[tidx,:,:]
    yt = yt[tidx]

    train_x, train_y = [
        X.view(X.shape[0],-1).float(),
        y.long()
    ]
    test_x, test_y = [
        Xt.view(Xt.shape[0],-1).float(),
        yt.long()
    ]
    return train_x, train_y, test_x, test_y, Xt

def imshow(img, label):
    plt.imshow(img.reshape((28,28)))
    plt.title(label)
    plt.show()

def custom_normalization(data, std, mean):
   return (data - mean) / std

def incremental_data(train_x, y, idx, model, thd=0.5, k=300):
    yp = model(train_x)
    yp.detach_()
    yp = yp.numpy()
    ypt = np.where(yp>thd)[0]
    yt = np.where(y==0)[0]
    ridx = np.intersect1d(ypt, yt)
    print("Correct prediction: {}/{}\n".format(ridx.shape[0], ypt.shape[0]))
    widx = np.array([i for i in ypt if i not in ridx ])
    ridx = np.setdiff1d(ridx, idx)
    widx = np.setdiff1d(widx, idx)
    seed(42)
    ridx = choice(ridx, size=2*k, replace=False)
    seed(42)
    widx = choice(widx, size=k, replace=False)
    idx = np.hstack((ridx, widx))
    seed(42)
    np.random.shuffle(idx)
    return idx

def GHM(y_pred, y, epsilon=0.1):
    batch_size =len(y)
    g = abs(y_pred-y)
    hist, bin_edges = np.histogram(g, bins = 10, range=[0,1])

    GD = hist/epsilon
    beta = []
    for g_i in g:
        if g_i == 0: # MLiao: the index of this case is -1.. 
            GD_gi = GD[0]
        else:
            GD_gi = GD[np.where(g_i <= bin_edges)[ 0][0]-1]
        beta.append(batch_size/GD_gi)
    return beta

def loss_batch(model, loss_func, x, y, opt=None):
    x = x.view(x.shape[0], -1)
    p_risk = model(x)
    loss = loss_func(p_risk, y)  

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, early_stopping=None, isplot=None):
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
        print("epoches= {},loss is {}".format(epoch, val_loss))

        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    idx = np.argmin(loss_v)
    print('minimum: ')
    print('epoch: {}, loss: {}'.format(idx, loss_v[idx]))
    if isplot != None:
        plt.plot(np.arange(len(loss_v)), np.array(loss_v),c='y')
        plt.plot(idx, loss_v[idx], marker='o')
        plt.title('loss-dnn')
        plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))
    return idx

def statistics(X, alpha=0.95):
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    SE = sig/np.sqrt(X.shape[0])
    CI = stats.norm.interval(alpha, loc=mu, scale=SE)
    return mu, sig, CI[0], CI[1]