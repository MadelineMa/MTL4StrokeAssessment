from shutil import get_archive_formats
import torch
import numpy as np
from torch import nn
from torch import optim
from sklearn.metrics import classification_report
from early_stopping import EarlyStopping
from utils import get_data, fit
from model import DeepQI

batch_size = 256

train_loader, valid_loader, test_x, test_y = get_data(batch_size)

# parameter set
lr = 1e-2
epochs = 30

feature_idx = [433, 349, 350]
num_dims = [784, 300, 100, 10]
model = DeepQI(feature_idx, num_dims)

loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr,)

patience = 10
early_stopping = EarlyStopping(patience, verbose=False)

fit(epochs, model, loss_func, opt, train_loader, valid_loader, early_stopping)
y_prob = model(test_x)
y_prob = y_prob.detach().numpy()
y_pred = np.argmax(y_prob.tolist(), axis=1)
report = classification_report(test_y, y_pred, digits=4)
print(report)