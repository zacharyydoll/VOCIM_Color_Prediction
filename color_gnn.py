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
            nn.Linear(hidden_dim * 2, hidden_dim),
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
    
    def forward(self, x, edge_index):
        # Edge update
        edge_attr = self.edge_model(x, edge_index)
        
        # Node update
        x = self.node_model(x, edge_index, edge_attr)
        
        return x, edge_attr

class ColorGNN(nn.Module):
    def __init__(self, num_colors, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        
        # Initial node feature projection
        self.node_proj = nn.Linear(num_colors, hidden_dim)  # Project TinyViT probabilities
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            ColorGNBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final projection for color assignment
        # Now outputs (num_birds, num_colors) directly
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
    
    def forward(self, probs):
        """
        Process a batch of birds from the same frame.
        Args:
            probs: Tensor of shape (num_birds, num_colors) containing TinyViT probabilities
        Returns:
            List of assigned color indices for each bird
        """
        # Ensure probs has the correct shape
        if len(probs.shape) != 2 or probs.shape[1] != self.num_colors:
            raise ValueError(f"Expected probs shape (num_birds, {self.num_colors}), got {probs.shape}")
        
        num_birds = probs.shape[0]
        device = probs.device
        
        """
        Get top-K colors among TinyViT probabilities, which gives two matrices:
            - top_k_values: The actual probability values (num_birds x num_birds)
            - top_k_indices: The indices of these values in the original matrix (num_birds x num_birds), 
              to keep track of which color the top_K probabilities correspond to in the original matrix 
        """
        top_k_values, top_k_indices = torch.topk(probs, k=num_birds, dim=1)
        
        # create bipartite graph using only the top-K colors from TinyViT preds 
        filtered_probs = torch.zeros_like(probs)
        for i in range(num_birds):
            filtered_probs[i, top_k_indices[i]] = top_k_values[i]
        data = self.create_bipartite_graph(filtered_probs)
        
        x = data.x
        for gnn in self.gnn_layers: # process through gnn layers
            x, _ = gnn(x, data.edge_index)
            x = self.dropout(x)
        
        gnn_scores = self.color_proj(x[:num_birds]) #[nb_birds_in_frame x num_colors] gnn output 
                
        # Combine GNN scores with TinyViT probabilities by weighing gnn scores with the TinyViT probs.
        # i.e. low color confidence from TinyViT -> its gnn score will be weighed less, and vice-versa.
        combined_scores = torch.zeros(num_birds, self.num_colors, device=device)
        
        for i in range(num_birds):
            for j in range(self.num_colors):
                if j in top_k_indices[i]:  # only combine scores for top-K colors
                    gnn_score = gnn_scores[i, j]  
                    tinyvit_prob = probs[i, j]   
                    combined_scores[i, j] = gnn_score * tinyvit_prob
        
        # create square cost matrix for Hungarian algorithm (num_birds x num_birds)
        cost_matrix = torch.zeros(num_birds, num_birds, device=device)
        
        for i in range(num_birds):
            for j in range(num_birds):
                color_idx = top_k_indices[i, j] # find color idx for this position
                cost_matrix[i, j] = 1 - combined_scores[i, color_idx] # set cost (1 - probability), because Hungarian minimizes cost
        
        # Apply Hungarian algo to ensure bijective assignment per frame
        # Detach from computation graph before converting to numpy
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        
        # Map back to original color indices
        final_assignments = top_k_indices[torch.arange(num_birds, device=device), col_ind]
        
        return final_assignments 