import numpy as np
import torch
# from torch import optim
# from early_stopping import EarlyStopping

def loss_batch(model, loss_func, x, y, opt=None):
    """
    Batch based loss computation.
    """
    loss = loss_func(model(x), y)

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
        # if epoch % 10 == 0:
            # print(epoch, val_loss)
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt')) 