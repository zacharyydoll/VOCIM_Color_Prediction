# model
model_used = "tinyvit" # "tinyvit" or "resnet"

# settings
batch_size = 16 # was 16, increased temporarily for debugging
num_epochs = 50
dropout_rate = 0.3
learning_rate = 1e-04
weight_decay = 0.01

# Smoothing
smoothing = 0.1

# Mask settings
sigma_val=13 # was 13
use_heatmap_mask=True

# Scheduler params
scheduler_factor = 0.5
scheduler_patience = 3

# Model params
num_classes = 8
model_name = 'tiny_vit_21m_512.dist_in22k_ft_in1k'

# ColorGNN configuration
use_glan = True  # now the color GNN (kept GLAN name for backwards compatibility)
glan_hidden_dim = 256  # Hidden dimension for graph layers
glan_num_layers = 3  # Number of graph network blocks
glan_dropout = 0.1  # Dropout rate for graph layers
glan_lr = 2e-4 
glan_early_stop = 20 
glan_epochs = 100

# Evaluation metrics configuration
compute_confusion_matrix = True
compute_class_metrics = True
compute_roc_auc = True
compute_f1_score = True
compute_precision_recall = True
compute_graph_metrics = True