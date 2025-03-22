# check_mask.py
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import ImageDataset

def check_mask(json_path, img_dir):
    """
    Loads one sample from the dataset and visualizes the RGB image and
    the generated backpack mask (fourth channel) side-by-side.
    """
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img),  
        transforms.ToTensor()  
    ])
    
    dataset = ImageDataset(data_path=json_path, img_dir=img_dir, transform=transform)
    #print("Dataset length:", len(dataset))
    
    op_file = "check_mask_output.png"
    img_nb = 513 #10278 10279
    sample = dataset[img_nb]
    sample_anno = dataset.annotations[img_nb]
    #print(f'Annotation (index {img_nb}) from JSON:', sample_anno)
    #print("Bounding box (tensor):", sample['bbox'])
    #print("Bounding box (list):", sample['bbox'][0].tolist())

    img_id = sample_anno['image_id']
    img_info = dataset.img_paths[img_id]
    #print("Corresponding image entry:", img_info)

    image_with_mask = sample['image']  # expected shape: (4, H, W)
    
    expected_coord = sample_anno.get("backpack_coord")

    rgb = image_with_mask[:3].permute(1, 2, 0).numpy()  # shape (H, W, 3)
    mask = image_with_mask[3].numpy()  # shape (H, W)

    cmap = plt.cm.jet
    newcolors = cmap(np.linspace(0, 1, cmap.N))
   
    threshold = 0.2  # values below 20% of the colormap range become transparent
    cutoff = int(cmap.N * threshold)
    newcolors[:cutoff, -1] = 0  # set alpha=0 for the lower colors 
    newcmap = mcolors.ListedColormap(newcolors)

    # Create a single figure and overlay the mask on the RGB image.
    plt.figure(figsize=(10, 5))
    plt.imshow(rgb)  # show the original image

    # Overlay mask with colormap and set opacity alpha
    plt.imshow(mask, cmap=newcmap, alpha=0.2)  # adjust for opacity 
    plt.axis('off')
    plt.title("Image with Backpack Mask Overlay")
    plt.savefig(op_file)
    plt.show()

    print(f"image saved to {op_file}")

if __name__ == "__main__":
    check_mask("/mydata/vocim/zachary/data/cropped_merged_annotations.json", "/mydata/vocim/zachary/data/cropped")


