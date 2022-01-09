import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Stroke_LR(nn.Module):
    """
    Logistic regression returns the probablity of stroking.
    """
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(34,1)
    def forward(self, x):
        return torch.sigmoid(self.lin(x)) # change to F.sigmoid(self.lin(x))

# TODO: Stroke_DNN

class Risk_DNN(nn.Module):
    """
    DNN returns the a triplet including the probablities of `low, medium, high` risks.
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(35,17) # TODO, 34 -> 35 (append output of `Stroke_LR`)
        self.lin2 = nn.Linear(17, 3)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2)
        return x

class StrokeRiskModel(nn.Module):
    """
    stroke-risk model return the probability of stroking along with ones of `low, medium, high` risks.
    """
    def __init__(self):
        super().__init__()
        # LR model of stroking prediction.
        self.lin1 = nn.Linear(34, 1)
        # DNN model of ranking risks. (35 = 34 + the output of self.lin1)
        # self.lin2 = nn.Linear(35, 17)
        self.lin2 = nn.Linear(34, 17)
        self.lin3 = nn.Linear(17, 4)
    def forward(self, x):
        # Probablity of stroking
        ys = torch.sigmoid(self.lin1(x))
        # yr = torch.cat([ys, x], dim=1)
        # yr = F.relu(self.lin2(yr))
        yr = F.relu(self.lin2(x))
        yr = F.dropout(yr, p=0.2)
        yr = F.relu(self.lin3(yr))
        yr = F.dropout(yr, p=0.2)
        return ys, yr