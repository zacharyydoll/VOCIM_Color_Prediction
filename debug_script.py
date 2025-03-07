from dataloader import get_train_dataloder

# Load a small batch of data
train_loader = get_train_dataloder('data/vocim_yolopose_train_vidsplit.json', '/mydata/vocim/zachary/data/cropped', batch_size=2)

# Inspect the first batch
for batch in train_loader:
    images = batch['image']
    print("Batch image shape:", images.shape)  # Should be [batch_size, 3, 512, 512]
    break

    