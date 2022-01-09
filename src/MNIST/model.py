import torch
import copy
from torch import nn
from itertools import combinations

class DNN(nn.Module):
    def __init__(self, num_dims, num_classes):
        super().__init__()
        self.num_dims = num_dims
        nd = len(num_dims)
        self.nd = len(num_dims)
        self.num_classes = num_classes
        for i in range(nd-2):
            setattr(self, 'lin'+str(i),
                    nn.Linear(num_dims[i], num_dims[i+1]))
            setattr(self, 'bn'+str(i),
                    nn.BatchNorm1d(num_dims[i+1]))
            setattr(self, 'relu'+str(i),
                    nn.ReLU())
        self.output = nn.Linear(num_dims[nd-2], num_classes)
        if num_classes==1:
            self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        for i in range(self.nd-2):
            x = getattr(self, 'lin'+str(i))(x)
            x = getattr(self, 'bn'+str(i))(x)
            x = getattr(self, 'relu'+str(i))(x)
        x = self.output(x)
        if self.num_classes==1:
            x = self.sigmoid(x)
        return x

class DeepQI(nn.Module):
    def __init__(self, feature_idx, mpl_dims, drop_rate=0.2):
        super().__init__()
        self.field_size = len(feature_idx)
        self.feature_idx = feature_idx
        # self.embedding_size = embedding_size
        self.mpl_dims = mpl_dims
        nd = len(mpl_dims)
        self.nd = nd
        comb_fea_index = []
        for idx in combinations(range(self.field_size), 2):
            comb_fea_index.append(list(idx))
        ## QI part
        # self.embeddings = nn.ModuleList(
            # [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        # )
        self.comb_fea_index = comb_fea_index
        # dnn part
        self.lin0 = nn.Linear(mpl_dims[0] + len(comb_fea_index), mpl_dims[1])
        self.bn0 = nn.BatchNorm1d(mpl_dims[1])
        self.relu0 = nn.ReLU()
        # self.dropout0 = nn.Dropout(p=drop_rate)
        for i in range(1, nd-2):
            setattr(self, 'lin'+str(i),
                    nn.Linear(mpl_dims[i], mpl_dims[i+1]))
            setattr(self, 'bn'+str(i),
                    nn.BatchNorm1d(mpl_dims[i+1]))
            setattr(self, 'relu'+str(i),
                    nn.ReLU())
        self.output = nn.Linear(mpl_dims[-2], mpl_dims[-1])

    def forward(self, xv): #, xi):
        # embs = []
        # for i, emb in enumerate(self.embeddings):
        #     embs.append(emb(xi[:,i]))
        # embs = torch.stack(embs)
        qi = []
        for pair in self.comb_fea_index:
            # i, j = pair
            i, j = self.feature_idx[pair[0]], self.feature_idx[pair[1]]
            qi.append((xv[:,i] * xv[:,j]).unsqueeze(1))
        qi = torch.cat(qi, dim=1)
        qi = qi/255#/255
        xv = torch.cat((xv, qi), dim=1)

        for i in range(0, self.nd-2):
            xv = getattr(self, 'lin'+str(i))(xv)
            xv = getattr(self, 'bn'+str(i))(xv)
            xv = getattr(self, 'relu'+str(i))(xv)
        xv = self.output(xv)
        return xv

class MMOE(nn.Module):
    def __init__(self, feature_idx, qi_dims, dnn_dims):#, embedding_size=4, drop_rate=0.2):
        super().__init__()
        self.field_size = len(feature_idx)
        self.feature_idx = feature_idx
        # self.embedding_size = embedding_size
        self.qi_dims = qi_dims
        self.dnn_dims = dnn_dims
        nd = len(qi_dims)
        self.nd = nd
        comb_fea_index = []
        for idx in combinations(range(self.field_size), 2):
            comb_fea_index.append(list(idx))
        ## QI part
        # self.embeddings = nn.ModuleList(
        #     [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        # )
        self.comb_fea_index = comb_fea_index
        # QI-dnn part
        self.QIlin0 = nn.Linear(qi_dims[0] + len(comb_fea_index), qi_dims[1])
        self.QIbn0 = nn.BatchNorm1d(qi_dims[1])
        self.QIrelu0 = nn.ReLU()
        for i in range(1, nd-2):
            setattr(self, 'QIlin'+str(i),
                    nn.Linear(qi_dims[i], qi_dims[i+1]))
            setattr(self, 'QIbn'+str(i),
                    nn.BatchNorm1d(qi_dims[i+1]))
            setattr(self, 'QIrelu'+str(i),
                    nn.ReLU())
        self.QIout = nn.Linear(qi_dims[-2], qi_dims[-1])
        # pure dnn part
        for i in range(len(dnn_dims)-2):
            setattr(self, 'DNNlin'+str(i),
                    nn.Linear(dnn_dims[i], dnn_dims[i+1]))
            setattr(self, 'DNNbn'+str(i),
                    nn.BatchNorm1d(dnn_dims[i+1]))
            setattr(self, 'DNNrelu'+str(i),
                    nn.ReLU())
        self.DNNOut = nn.Linear(dnn_dims[-2], dnn_dims[-1])
        self.sigmoid = nn.Sigmoid()
        for i in range(1, 3): # number of towers, fixed to 2.
            setattr(self, 'GATE'+str(i)+'lin',
                    nn.Linear(dnn_dims[0], 2))
            setattr(self, 'GATE'+str(i)+'softmax',
                    nn.Softmax(dim=1))   

    def forward(self, xv):
        # QI part
        # embs = []
        # for i, emb in enumerate(self.embeddings):
        #     embs.append(emb(xi[:,i]))
        # embs = torch.stack(embs)
        qi = []
        for pair in self.comb_fea_index:
            i, j = self.feature_idx[pair[0]], self.feature_idx[pair[1]]
            qi.append((xv[:,i] * xv[:,j]).unsqueeze(1))
        qi = torch.cat(qi, dim=1)
        qi = qi/255#/255
        qi = torch.cat((xv, qi), dim=1)

        for i in range(0, self.nd-3):
            qi = getattr(self, 'QIlin'+str(i))(qi)
            qi = getattr(self, 'QIbn'+str(i))(qi)
            qi = getattr(self, 'QIrelu'+str(i))(qi)

        # pure dnn part
        xdnn = copy.deepcopy(xv)
        for i in range(0, len(self.dnn_dims)-3):
            xdnn = getattr(self, 'DNNlin'+str(i))(xdnn)
            xdnn = getattr(self, 'DNNbn'+str(i))(xdnn)
            xdnn = getattr(self, 'DNNrelu'+str(i))(xdnn)
        # gate
        x = torch.stack([torch.rand(xdnn.shape), torch.rand(qi.shape)])
        for i in range(1, 3):
            g = getattr(self, 'GATE'+str(i)+'lin')(xv)
            g = getattr(self, 'GATE'+str(i)+'softmax')(g)
            x[i-1] = g[:,0].unsqueeze(dim=1)*qi + g[:,1].unsqueeze(dim=1)*xdnn
        # gate offers weights to connect QI and DNN
        ii = self.nd-3
        xqi = getattr(self, 'QIlin'+str(ii))(x[0])
        xqi = getattr(self, 'QIbn'+str(ii))(xqi)
        xqi = getattr(self, 'QIrelu'+str(ii))(xqi)
        xqi = self.QIout(xqi)
        ii = len(self.dnn_dims)-3
        xdnn = getattr(self, 'DNNlin'+str(ii))(x[1])
        xdnn = getattr(self, 'DNNbn'+str(ii))(xdnn)
        xdnn = getattr(self, 'DNNrelu'+str(ii))(xdnn)
        xdnn = self.DNNOut(xdnn)
        xdnn = self.sigmoid(xdnn)
        return xdnn, xqi