import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(SAGEConv(sizes[i], sizes[i + 1]))
            if i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(sizes[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def decode_all(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.t())
