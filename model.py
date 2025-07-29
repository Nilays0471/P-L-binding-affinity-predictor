import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, in_c, hid):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, 1)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.conv1(x, ei).relu()
        x = self.conv2(x, ei).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)
