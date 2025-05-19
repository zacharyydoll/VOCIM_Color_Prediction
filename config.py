# model
model_used = "tinyvit" # "tinyvit" or "resnet"

# settings
batch_size = 16 # was 16, increased temporarily for debugging
num_epochs = 50
dropout_rate = 0.3
learning_rate = 1e-04 # transformer learning rate 
weight_decay = 0.01
freeze_tinyvit=False # freeze or unfreeze last block of tinyvit 

# Smoothing
smoothing = 0.1

# Mask settings
sigma_val=13 # was 13
use_heatmap_mask=True

# Scheduler params
scheduler_factor = 0.5
scheduler_patience = 3
scheduler_mode = 'max' # or min 

# Model params
num_classes = 8
model_name = 'tiny_vit_21m_512.dist_in22k_ft_in1k'

# ColorGNN configuration
use_glan = True  # now the color GNN (kept GLAN name for backwards compatibility)
glan_hidden_dim = 256  
glan_num_layers = 4  # previously was 4 -> 97.95% acc. TODO: try with 3 or 5 
glan_dropout = 0.1  # already tried 0.2 -> worse results 
glan_lr = 5e-4 # TODO: already tried with 2e-4, 5e-5 -> <97% acc
glan_early_stop = 25    
glan_epochs = 80
glan_weight_decay = 0.01 # was 1e-4 -> ~96% acc stabilization

# Evaluation metrics configuration
compute_confusion_matrix = True
compute_class_metrics = True
compute_roc_auc = True
compute_f1_score = True
compute_precision_recall = True
compute_graph_metrics = True