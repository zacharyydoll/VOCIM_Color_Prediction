# model
model_used = "resnet" # or "tinyvit"

# settings
batch_size = 16
num_epochs = 50
dropout_rate = 0.3
learning_rate = 1e-04
weight_decay = 0.01

# Smoothing
smoothing = 0.1

# Mask settings
sigma_val=13 # was 13
use_heatmap_mask=False

# Scheduler params
scheduler_factor = 0.5
scheduler_patience = 3

# Model params
num_classes = 8
model_name = 'tiny_vit_21m_512.dist_in22k_ft_in1k'

#TODO: increase dropout from 0.5 to 0.6, weight decay to 0.03, scheduler patience to 4


# TODO: write a dictionnary to collect the files with 2 or more backpacks in the same image, 
# then keep track of how often 