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
    
    def create_bipartite_graph(self, embeddings, probs):
        """
        Create a bipartite graph for a single frame's birds.
        Args:
            embeddings: Tensor of shape (num_birds, embedding_dim) from TinyViT
            probs: Tensor of shape (num_birds, num_colors) containing TinyViT probabilities
        Returns:
            Data object containing the bipartite graph
        """
        num_birds = embeddings.shape[0]
        device = embeddings.device
        # Bird node features: TinyViT embeddings
        bird_nodes = embeddings  # shape: (num_birds, embedding_dim)
        # Color node features: one-hot, projected to embedding_dim
        color_nodes = torch.eye(self.num_colors, device=device)  # (num_colors, num_colors)
        # Optionally project color nodes to embedding_dim if needed
        if color_nodes.shape[1] != bird_nodes.shape[1]:
            color_nodes = F.pad(color_nodes, (0, bird_nodes.shape[1] - color_nodes.shape[1]))
        x = torch.cat([bird_nodes, color_nodes], dim=0)
        # Edge connections (all birds to all colors)
        bird_indices = torch.arange(num_birds, device=device)
        color_indices = torch.arange(self.num_colors, device=device) + num_birds
        bird_idx = bird_indices.repeat_interleave(self.num_colors)
        color_idx = color_indices.repeat(num_birds)
        edge_index = torch.stack([bird_idx, color_idx], dim=0)
        # Edge attributes: TinyViT probabilities for each bird-color pair
        edge_attr = probs.view(-1, 1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def forward_combined(self, embeddings, probs):
        data = self.create_bipartite_graph(embeddings, probs)
        x = data.x
        edge_idx = data.edge_index
        edge_attr = self.edge_proj(data.edge_attr)  # project edge attributes
        for block in self.gnn_layers:
            x, edge_attr = block(x, edge_idx, edge_attr)
            x = self.dropout(x)
        bird_scores = self.color_proj(x[:embeddings.size(0)])  # only the bird-nodes
        return bird_scores * probs

    def assign(self, embeddings, probs, tinyvit_probs=None):
        combined = self.forward_combined(embeddings, probs)   # [N_birds × C_colors]
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)
        combined = combined.clamp(0.0, 1.0)
        N_birds, N_colors = combined.shape
        if tinyvit_probs is not None:
            if isinstance(tinyvit_probs, torch.Tensor):
                tinyvit_probs_np = tinyvit_probs.detach().cpu().numpy()
            else:
                tinyvit_probs_np = np.array(tinyvit_probs)
            color_sums = tinyvit_probs_np.sum(axis=0)
            topk = N_birds
            topk_color_indices = np.argsort(color_sums)[-topk:][::-1]
            combined_np = combined.detach().cpu().numpy()
            reduced_combined = combined_np[:, topk_color_indices]
            reduced_combined = np.ascontiguousarray(reduced_combined)
            cost = (1.0 - reduced_combined)
            row, col = linear_sum_assignment(cost)
            assigned_colors = [topk_color_indices[j] for j in col]
            return torch.tensor(assigned_colors, device=embeddings.device)
        else:
            cost = (1.0 - combined).detach().cpu().numpy()
            row, col = linear_sum_assignment(cost)
            return torch.tensor(col, device=embeddings.device)       