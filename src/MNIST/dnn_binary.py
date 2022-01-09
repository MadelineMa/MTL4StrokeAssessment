from shutil import get_archive_formats
import torch
from torch import nn
from torch import optim
from numpy.random import choice, seed
from sklearn.metrics import classification_report
from early_stopping import EarlyStopping
from utils import get_data_binary, fit
from model import DNN

batch_size = 256

train_loader, valid_loader, test_x, test_y = get_data_binary(batch_size)

# parameter set
lr = 1e-2
epochs = 200
num_dims = [784, 300, 100, 1]
model = DNN(num_dims, 1)

loss_func = nn.BCELoss()
opt = optim.SGD(model.parameters(), lr=lr)

patience = 10
early_stopping = EarlyStopping(patience, verbose=False)

fit(epochs, model, loss_func, opt, train_loader, valid_loader, early_stopping)
y_prob = model(test_x)
y_prob = y_prob.detach().numpy()
y_pred = [1 if y_prob[i] > 0.5 else 0 for i in range(len(y_prob))]
report = classification_report(test_y, y_pred, digits=4)
print(report)