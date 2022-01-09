import numpy as np
import torch
import sys
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.metrics import classification_report
from strokeDataset import StrokedDataset
# from strokeModels import Stroke_LR, Risk_DNN
from strokeModels import StrokeRiskModel
# from strokeUtils import loss_batch, fit
from early_stopping import EarlyStopping

epsilon = torch.tensor(sys.float_info.epsilon)

def loss_strok_high(prob_strok, prob_high): # type of input should tensor
    ps = prob_strok.squeeze()
    return 1 - (ps*prob_high + epsilon)/(prob_high + epsilon)

def loss_strok_low(prob_strok, prob_low):
    ps = prob_strok.squeeze()
    return (ps*prob_low + epsilon)/(prob_low + epsilon)

def loss_batch(model, ls, lr, x, ys, yr, opt=None):
    """
    Batch based loss computation.
    """
    # construct moop loss
    p_stroke, p_risk = model(x)
    lsh = loss_strok_high(p_stroke, p_risk[:,2])
    lsl = loss_strok_low(p_stroke, p_risk[:,0])
    loss = ls(p_stroke, ys) + lr(p_risk, yr) + lsl.sum()/len(x) #+ 0.2 * lsh.sum()/len(x) + 0.2 * lsl.sum()/len(x)
    # loss = lr(p_risk, yr)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)

def fit(epochs, model, ls, lr, opt, train_dl, valid_dl, early_stopping=None):
    for epoch in range(epochs):
        model.train()
        for xb, ysb, yrb in train_dl:
            loss_batch(model, ls, lr, xb, ysb, yrb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, ls, lr, xb, ysb, yrb) for xb, ysb, yrb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

        if early_stopping is not None:  
            early_stopping(val_loss, model) # take care of this part
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # if epoch % 10 == 0:
            # print(epoch, val_loss)
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt')) 

if __name__ == '__main__':
    # parameter set #TODO invoke argparse?
    if os.name == 'posix':
        train_file = '../GN-data/train.csv'
        valid_file = '../GN-data/valid.csv'
        test_file = '../GN-data/test.csv'
    else:
        train_file = '..\\GN-data\\train.csv'
        valid_file = '..\\GN-data\\valid.csv'
        test_file = '..\\GN-data\\test.csv'
    lr = 0.001
    bs = 100
    epochs = 400

    train_ds = StrokedDataset(train_file)
    valid_ds = StrokedDataset(valid_file, train_ds.scaler)
    test_ds = StrokedDataset(test_file, train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)

    model = StrokeRiskModel()
    # output_risk = risk_model(torch.cat([output_stroke, train_ds], dim=1)) # or dim = 0?
    Ls = nn.BCELoss() # Loss of the stroking sub-model
    Lr = nn.CrossEntropyLoss() # Loss of the risk sub-model
    # Lsh = loss_strok_high(output_stroke, output_risk[2]) # check
    # Lsl = loss_strok_low(output_stroke, output_risk[0])

    # Loss = Ls + Lr + Lsh + Lsl
    # Original run_dnn for reference
    # model = Stroke_DNN_Model()
    # loss_func = F.cross_entropy
    # loss_func = F.binary_cross_entropy
    # loss_func = nn.CrossEntropyLoss()
    # opt = optim.RMSprop([stroke_model.parameters(), risk_model.parameters()], lr=lr) # The optimizer need further consideration
    opt = optim.RMSprop(model.parameters(), lr=lr)

    patience = 20	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)
# def fit(epochs, stroke_model, risk_model, ls, lr, opt, train_dl, valid_dl, early_stopping=None):
    fit(epochs, model, Ls, Lr, opt, train_dl, test_dl, early_stopping)
    ys_prob, yr_prob = model(test_ds.data)
    ys_prob = ys_prob.squeeze().tolist() # MLiao: squeeze may not needed
    ys_pred = np.array([0 if ys_prob[i] < 0.5 else 1 for i in range(len(ys_prob))])
    ys_test = test_ds.stroke_labels.tolist()
    strok_report = classification_report(ys_test, ys_pred, digits=4)
    print(strok_report)
    # report of risk ranking
    yr_pred = np.argmax(yr_prob.tolist(), axis=1)
    yr_test = test_ds.risk_labels.tolist()
    risk_report = classification_report(yr_test, yr_pred, digits=4)
    print(risk_report)