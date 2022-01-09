import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")
class StrokedDataset(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = scaler
        data, risk_labels, stroke_labels = self.dataframe, self.dataframe.pop('风险评级'), self.dataframe.pop('卒中')
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        # self.labels = torch.tensor(labels).float() # lr
        self.stroke_labels = torch.tensor(stroke_labels).float()
        self.risk_labels = torch.tensor(risk_labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        stroke_labels = self.stroke_labels
        risk_labels = self.risk_labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], stroke_labels[idx].unsqueeze(dim=-1), risk_labels[idx]

class DnnDataset(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = scaler
#        data, labels = self.dataframe, self.dataframe.pop('风险评级')
        self.dataframe.pop('Apx')
        data, labels = self.dataframe, self.dataframe.pop('RR')
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        # self.labels = torch.tensor(labels).float() # lr
        # self.stroke_labels = torch.tensor(stroke_labels).float()
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        # stroke_labels = self.stroke_labels
        labels = self.labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], labels[idx]

class Dnn20(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = scaler
#        data, labels = self.dataframe, self.dataframe.pop('风险评级')
        # self.dataframe.pop('Apx')
        data, labels = self.dataframe, self.dataframe.pop('RR')
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        # self.labels = torch.tensor(labels).float() # lr
        # self.stroke_labels = torch.tensor(stroke_labels).float()
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        # stroke_labels = self.stroke_labels
        labels = self.labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], labels[idx]
        
class QIDataset(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, idx_file, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.index = pd.read_csv(idx_file)
        # assert self.dataframe.shape == self.df_index.shape
        self.scaler = scaler
#        data, labels = self.dataframe, self.dataframe.pop('风险评级')
        self.dataframe.pop('Apx') # uncomment for QI_category
        data, labels = self.dataframe, self.dataframe.pop('RR')
        self.index.pop('RR')
        self.index.pop('Apx') # uncomment for QI_category
        # index = self.df_index
        
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        self.index = torch.tensor(self.index.values)
        assert self.data.shape == self.index.shape
        # self.labels = torch.tensor(labels).float() # lr
        # self.stroke_labels = torch.tensor(stroke_labels).float()
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        index = self.index
        # stroke_labels = self.stroke_labels
        labels = self.labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], index[idx], labels[idx]

class QI20(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, idx_file, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.index = pd.read_csv(idx_file)
        # assert self.dataframe.shape == self.df_index.shape
        self.scaler = scaler
#        data, labels = self.dataframe, self.dataframe.pop('风险评级')
        # self.dataframe.pop('Apx') # uncomment for QI_category
        data, labels = self.dataframe, self.dataframe.pop('RR')
        self.index.pop('RR')
        # self.index.pop('Apx') # uncomment for QI_category
        # index = self.df_index
        
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        self.index = torch.tensor(self.index.values)
        assert self.data.shape == self.index.shape
        # self.labels = torch.tensor(labels).float() # lr
        # self.stroke_labels = torch.tensor(stroke_labels).float()
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        index = self.index
        # stroke_labels = self.stroke_labels
        labels = self.labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], index[idx], labels[idx]

class BinaryDataset(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = scaler
        data, labels = self.dataframe, self.dataframe.pop('Apx')
        self.dataframe.pop('RR')
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        # self.labels = torch.tensor(labels).float() # lr
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        labels = self.labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], labels[idx].unsqueeze(dim=-1)

class MoEDataset(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, idx_file, scaler=None):
        """
            csv_file (String): data file to be read.
            idx_file (String): index file corresponding to csv_file
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.index = pd.read_csv(idx_file)
        # assert self.dataframe.shape == self.df_index.shape
        self.scaler = scaler
        # data, labels = self.dataframe, self.dataframe.pop('风险评级')
        data, rlabels, slabels = self.dataframe, self.dataframe.pop('RR'), self.dataframe.pop('Apx')
        self.index.pop('RR')
        self.index.pop('Apx')
        # index = self.df_index
        
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        self.index = torch.tensor(self.index.values)
        assert self.data.shape == self.index.shape
        # self.labels = torch.tensor(labels).float() # lr
        self.slabels = torch.tensor(slabels).float()
        self.rlabels = torch.tensor(rlabels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        index = self.index
        # stroke_labels = self.stroke_labels
        slabels = self.slabels
        rlabels = self.rlabels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], index[idx], slabels[idx].unsqueeze(-1), rlabels[idx]
