import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import yaml  
from config import sigma_val, use_heatmap_mask

def create_heatmap_mask(image_size, center, sigma=sigma_val):
    """
    Creates a Gaussian heatmap mask of shape (H, W) where the maximum (1.0)
    is at the given center.
    """
    H, W = image_size
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    heatmap = np.exp(-((xv - center[0]) ** 2 + (yv - center[1]) ** 2) / (2 * sigma ** 2))
    return torch.tensor(heatmap, dtype=torch.float32)



class ImageDataset(Dataset):
    def __init__(self, data_path, img_dir, transform=None, use_mask=use_heatmap_mask, mask_sigma=sigma_val,
                 bird_identity_yaml="/mydata/vocim/zachary/color_prediction/newdata_bird_identity.yaml",
                 colormap_yaml="/mydata/vocim/zachary/color_prediction/newdata_colormap.yaml"):
        """
        Args:
            data_path (str): Path to the JSON annotations file.
            img_dir (str): Directory containing the images.
            transform (callable, optional): Transform to be applied to the image.
            mask_sigma (float): Sigma to be used in the heatmap mask.
            bird_identity_yaml (str): Path to the YAML file mapping image directories to bird identities.
            colormap_yaml (str): Path to the YAML file mapping color names to numeric labels.
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.img_dir = img_dir
        self.img_paths = data['images']
        self.annotations = data['annotations']
        self.transform = transform
        self.mask_sigma = mask_sigma
        self.use_mask = use_mask

        # load YAML color mappings
        with open(bird_identity_yaml, "r") as f:
            self.bird_identity_mapping = yaml.safe_load(f)
        with open(colormap_yaml, "r") as f:
            self.color_map = yaml.safe_load(f)

    def __len__(self):
        return len(self.annotations)

    def get_effective_label(self, annotation):
        identity_str = annotation['identity']
        m = re.search(r'(bird(?:_[a-z])?_) *(\d+)', identity_str, re.IGNORECASE)
        if m:
            bird_key = f"bird_{m.group(2)}"
        else:
            raise ValueError(f"Could not parse identity from: {identity_str}")
        
        file_name = self.img_paths[annotation['image_id']]['file_name']
        directory = os.path.normpath(os.path.dirname(file_name))
        
        if directory not in self.bird_identity_mapping:
            # fallback: try basename
            alt_directory = os.path.basename(directory)
            if alt_directory in self.bird_identity_mapping: 
                directory = alt_directory
            else:
                raise ValueError(f"Directory {directory} not found in bird identity mapping.")
        
        bird_mapping = self.bird_identity_mapping[directory]
        
        if bird_key not in bird_mapping:
            raise ValueError(f"Bird key {bird_key} not found for directory {directory}.")
        color_name = bird_mapping[bird_key]
        
        if color_name not in self.color_map:
            raise ValueError(f"Color {color_name} not found in colormap.")
        label = self.color_map[color_name]

        #print(f"Annotation identity: {identity_str}")
        #print(f"Normalized directory: {directory}")
        #print(f"Fallback directory: {os.path.basename(directory)}")

        return label

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        label = self.get_effective_label(annotation)

        img_idx = annotation['image_id']
        file_name = self.img_paths[img_idx]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        assert os.path.exists(img_path), f"Image doesn't exist at {img_path}"

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # expected shape: (3, H, W)
        
        if self.use_mask: 
            _, H, W = image.shape

            # Get backpack coordinate from image info if available
            img_info = self.img_paths[img_idx]
            backpack_coord = img_info.get("backpack_coord", None)
            if backpack_coord is None:
                backpack_coord = (W / 2, H / 2) # should never happen since dataset mod 
            else:
                backpack_coord = (float(backpack_coord[0]), float(backpack_coord[1]))

            # Create heatmap mask using the provided backpack coordinate.
            mask = create_heatmap_mask((H, W), backpack_coord, sigma=self.mask_sigma)
            mask = mask.unsqueeze(0)
            
            # concatenate mask with the image to form a 4-channel input
            image_with_mask = torch.cat([image, mask], dim=0) 
        else: image_with_mask = image 

        x, y, w, h = annotation['bbox']
        sample = {
            'image': image_with_mask,
            'label': label,
            'image_path': img_path,
            'bbox': [torch.tensor([x, y, w, h])]
        }
        return sample
