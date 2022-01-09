# from shutil import get_archive_formats
import torch
import os
import numpy as np
from sklearn.metrics import classification_report
from utils import get_data
from model import DNN

dnnDir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'posix':
   ptDir = os.path.join(dnnDir, "pt/")
   npzDir = os.path.join(dnnDir, "npz/")
else:
   ptDir = os.path.join(dnnDir, "pt\\")
   npzDir = op.path.join(dnnDir, "npz\\")

batch_size = 256

train_loader, valid_loader, test_x, test_y = get_data(batch_size)

# parameter set
lr = 1e-2
epochs = 30

num_dims = [784, 300, 100, 10]
model = DNN(num_dims, 10)
ptFn = os.path.join(ptDir, "dnn.pt")
model.load_state_dict(torch.load(ptFn))
y_prob = model(test_x)
y_prob = y_prob.detach().numpy()
y_pred = np.argmax(y_prob.tolist(), axis=1)
report = classification_report(test_y, y_pred, digits=4)
print(report)