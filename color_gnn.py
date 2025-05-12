import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from scipy.optimize import linear_sum_assignment
import numpy as np
import re

def extract_frame_id(filename):
    """Extract frame ID from filename (e.g., 'img00332' from 'img00332_bird_1.png')"""
    match = re.search(r'img(\d+)_bird', filename)
    return match.group(1) if match else None

class ColorEdgeModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
         )
    
    def forward(self, x, edge_index):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        return self.edge_mlp(edge_features)

class ColorNodeModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Aggregate edge features for each node
        edge_aggr = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        edge_aggr.index_add_(0, row, edge_attr)
        edge_aggr.index_add_(0, col, edge_attr)
        
        # Combine node and edge features
        node_features = torch.cat([x, edge_aggr], dim=1)
        return self.node_mlp(node_features)

class ColorGNBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_model = ColorEdgeModel(hidden_dim)
        self.node_model = ColorNodeModel(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # edge_attr has shape [#edges, 1] (your TinyViT probs)
        row, col = edge_index
        # concatenate node features + existing edge_attr
        edge_input = torch.cat([ x[row], x[col], edge_attr ], dim=1)
        edge_attr = self.edge_model.edge_mlp(edge_input)   # [#edges, hidden_dim]
        
        # node update
        x = self.node_model(x, edge_index, edge_attr)
        return x, edge_attr

class ColorGNN(nn.Module):
    def __init__(self, num_colors, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        
        # Initial node feature projection
        self.node_proj = nn.Linear(num_colors, hidden_dim)  # Project TinyViT probabilities
        self.edge_proj = nn.Linear(1, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            ColorGNBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final projection for color assignment
        self.color_proj = nn.Linear(hidden_dim, num_colors)
        self.dropout = nn.Dropout(dropout)
    
    def create_bipartite_graph(self, probs):
        """
        Create a bipartite graph for a single frame's birds.
        Args:
            probs: Tensor of shape (num_birds, num_colors) containing TinyViT probabilities
        Returns:
            Data object containing the bipartite graph
        """
        num_birds = probs.shape[0]
        device = probs.device
        
        # Create node features
        bird_nodes = self.node_proj(probs)  # Project TinyViT probabilities
        color_nodes = torch.eye(self.num_colors, device=device)  # One-hot encoding of colors
        color_nodes = self.node_proj(color_nodes)  # Project color nodes to same dimension
        x = torch.cat([bird_nodes, color_nodes], dim=0)
        
        # Create edge connections (all birds connect to all colors)
        bird_indices = torch.arange(num_birds, device=device)
        color_indices = torch.arange(self.num_colors, device=device) + num_birds
        
        # Create all possible bird-color pairs
        bird_idx = bird_indices.repeat_interleave(self.num_colors)
        color_idx = color_indices.repeat(num_birds)
        
        edge_index = torch.stack([bird_idx, color_idx], dim=0)
        
        # Edge attributes are the TinyViT probabilities
        edge_attr = probs.view(-1, 1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def forward_combined(self, probs):
        data = self.create_bipartite_graph(probs)
        x = data.x
        edge_idx = data.edge_index
        edge_attr = self.edge_proj(data.edge_attr)  

        # message-passing
        for block in self.gnn_layers:
            x, edge_attr = block(x, edge_idx, edge_attr)
            x = self.dropout(x)

        bird_scores = self.color_proj(x[:probs.size(0)])  # only the bird-nodes
        return bird_scores * probs

    def assign(self, probs):
        # at inference do Hungarian on combined scores
        combined = self.forward_combined(probs)   # [N_birds × C_colors]

        # 2) sanitize: replace NaN→0, +Inf→1, –Inf→0
        combined = torch.nan_to_num(combined,
                                   nan=0.0,
                                   posinf=1.0,
                                   neginf=0.0)
        # 3) clamp into [0,1] just in case you got tiny numerical overshoots
        combined = combined.clamp(0.0, 1.0)
        
        # 4) build cost matrix and Hungarian
        cost = (1.0 - combined).detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        return torch.tensor(col, device=probs.device)       