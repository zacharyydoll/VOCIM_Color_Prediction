import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from dataset import ImageDataset
from dataloader import letterbox

def check_letterboxed_mask(json_path, img_dir, sample_idx=0):
    """
    Loads one sample from the dataset with the full training transform (including letterboxing)
    and visualizes the RGB image and the generated backpack mask (fourth channel).
    This shows exactly what the model sees during training.
    """
    # Use the same transform as training
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: letterbox(img, size=(512, 512))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(data_path=json_path, img_dir=img_dir, transform=train_transform, use_mask=True)
    
    sample = dataset[sample_idx]
    sample_anno = dataset.annotations[dataset.valid_indices[sample_idx]]
    
    img_id = sample_anno['image_id']
    img_info = dataset.img_paths[img_id]
    
    print(f"Sample {sample_idx}:")
    print(f"Original image size: {img_info['width']}x{img_info['height']}")
    print(f"Backpack coordinates: {img_info.get('backpack_coord', 'None')}")
    print(f"Final image shape: {sample['image'].shape}")
    
    image_with_mask = sample['image']  # shape: (4, 512, 512)
    
    # Denormalize the RGB channels
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = image_with_mask[:3] * std + mean  # Denormalize
    rgb = torch.clamp(rgb, 0, 1)  # Clamp to [0, 1]
    rgb = rgb.permute(1, 2, 0).numpy()  # shape: (512, 512, 3)
    
    mask = image_with_mask[3].numpy()  # shape: (512, 512)
    
    # Create colormap for mask visualization
    cmap = plt.cm.jet
    newcolors = cmap(np.linspace(0, 1, cmap.N))
    threshold = 0.2
    cutoff = int(cmap.N * threshold)
    newcolors[:cutoff, -1] = 0
    newcmap = mcolors.ListedColormap(newcolors)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original RGB image (letterboxed)
    axes[0].imshow(rgb)
    axes[0].set_title("Letterboxed RGB Image (512x512)")
    axes[0].axis('off')
    
    # Mask only
    axes[1].imshow(mask, cmap='jet')
    axes[1].set_title("Backpack Mask (4th channel)")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(rgb)
    axes[2].imshow(mask, cmap=newcmap, alpha=0.3)
    axes[2].set_title("Mask Overlay on Letterboxed Image")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("letterboxed_mask_overlay.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to letterboxed_mask_overlay.png")
    
    # Print some statistics
    print(f"\nMask statistics:")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask min/max: {mask.min():.3f}/{mask.max():.3f}")
    print(f"Mask mean: {mask.mean():.3f}")
    print(f"Mask std: {mask.std():.3f}")
    
    # Find the peak of the mask
    peak_y, peak_x = np.unravel_index(np.argmax(mask), mask.shape)
    print(f"Mask peak at: ({peak_x}, {peak_y})")
    
    return sample

if __name__ == "__main__":
    # You can change the sample_idx to look at different examples
    sample = check_letterboxed_mask(
        json_path="/mydata/vocim/zachary/data/cropped_merged_annotations.json", 
        img_dir="/mydata/vocim/zachary/data/cropped",
        sample_idx=0  # Change this to look at different samples
    ) 