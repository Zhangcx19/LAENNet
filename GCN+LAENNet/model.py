import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid+nclass, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, nclass)

    def aggr_y(self, adj, labels):
        return torch.spmm(adj, labels)

    def forward(self, x, y_train, adj_x, adj_y):
        x1 = F.relu(self.gc1(x, adj_x))
        y = self.aggr_y(adj_y, y_train)
        x1 = torch.cat((x1, y.float()), -1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x = self.gc2(x1, adj_x)
        return F.log_softmax(x, dim=1), nn.Softmax(dim=1)(x), x