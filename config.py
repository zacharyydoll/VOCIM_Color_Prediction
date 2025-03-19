# settings
batch_size = 16
num_epochs = 50
dropout_rate = 0.3
learning_rate = 5e-05
weight_decay = 0.01

# Smoothing
smoothing = 0.1

# Mask settings
sigma_val=13

# Scheduler params
scheduler_factor = 0.5
scheduler_patience = 3

# Model params
num_classes = 8
model_name = 'tiny_vit_21m_512.dist_in22k_ft_in1k'

#TODO: increase dropout from 0.5 to 0.6, weight decay to 0.03, scheduler patience to 4
