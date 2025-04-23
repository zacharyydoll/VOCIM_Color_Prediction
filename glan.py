import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer

class EdgeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg = agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(1)), edge_attr)
        out = torch.cat([x, agg], dim=1)
        return self.node_mlp(out)

class GlobalModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        node_mean = torch.zeros_like(u)
        node_mean = node_mean.scatter_add_(0, batch.unsqueeze(-1).expand(-1, u.size(1)), x)
        counts = torch.zeros_like(u)
        counts = counts.scatter_add_(0, batch.unsqueeze(-1).expand(-1, u.size(1)), torch.ones_like(x))
        node_mean = node_mean / (counts + 1e-6)
        
        edge_mean = torch.zeros_like(u)
        edge_mean = edge_mean.scatter_add_(0, batch.unsqueeze(-1).expand(-1, u.size(1)), edge_attr)
        edge_mean = edge_mean / (counts + 1e-6)
        
        out = torch.cat([node_mean, edge_mean], dim=1)
        return self.global_mlp(out)

class GNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.gn = MetaLayer(
            edge_model=EdgeModel(node_dim, edge_dim, hidden_dim),
            node_model=NodeModel(node_dim, edge_dim, hidden_dim),
            global_model=GlobalModel(node_dim, edge_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.gn(x, edge_index, edge_attr, u, batch)

class GLAN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # Initial node and edge encoders
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # Graph network blocks
        self.gn_blocks = nn.ModuleList([
            GNBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode initial features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        u = torch.zeros((batch.max().item() + 1, x.size(1)), device=x.device)
        
        # Process through graph network blocks
        for i in range(self.num_layers):
            x, edge_attr, u = self.gn_blocks[i](x, edge_index, edge_attr, u, batch)
            
        # Global pooling and prediction
        x = torch.zeros_like(u)
        x = x.scatter_add_(0, batch.unsqueeze(-1).expand(-1, u.size(1)), x)
        counts = torch.zeros_like(u)
        counts = counts.scatter_add_(0, batch.unsqueeze(-1).expand(-1, u.size(1)), torch.ones_like(x))
        x = x / (counts + 1e-6)
        
        return self.pred_head(x).squeeze(-1) 