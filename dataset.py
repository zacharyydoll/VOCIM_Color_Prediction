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

def extract_frame_id(file_name):
    """Extract frame ID from filename (e.g., 'img00332_bird_1.png' -> 'img00332')"""
    match = re.match(r'(img\d+)_bird_\d+\.png', file_name)
    if match:
        return match.group(1)
    # Return default frame ID if pattern doesn't match
    return "unknown_frame"

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

        # Filter out invalid annotations
        self.valid_indices = []
        for idx, annotation in enumerate(self.annotations):
            try:
                self.get_effective_label(annotation)
                img_idx = annotation['image_id']
                file_name = self.img_paths[img_idx]['file_name']
                img_path = os.path.join(self.img_dir, file_name)
                if os.path.exists(img_path):
                    self.valid_indices.append(idx)
            except (ValueError, KeyError, AssertionError) as e:
                print(f"Warning: Skipping invalid annotation at index {idx}: {str(e)}")
                continue

    def __len__(self):
        return len(self.valid_indices)

    def get_effective_label(self, annotation):
        identity_str = annotation['identity']
        m = re.search(r'(bird(?:_[a-z])?_) *(\d+)', identity_str, re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not parse identity from: {identity_str}")
        
        bird_key = f"bird_{m.group(2)}"
        file_name = self.img_paths[annotation['image_id']]['file_name']
        directory = os.path.normpath(os.path.dirname(file_name))
        
        # find the directory in the mapping
        if directory not in self.bird_identity_mapping:
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
        
        return self.color_map[color_name]

    def __getitem__(self, idx):
        annotation = self.annotations[self.valid_indices[idx]]
        
        try:
            label = self.get_effective_label(annotation)
            img_idx = annotation['image_id']
            file_name = self.img_paths[img_idx]['file_name']
            img_path = os.path.join(self.img_dir, file_name)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image doesn't exist at {img_path}")

            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)  # expected shape: (3, H, W)
            
            if self.use_mask: 
                _, H, W = image.shape
                img_info = self.img_paths[img_idx]
                backpack_coord = img_info.get("backpack_coord", None)
                if backpack_coord is None:
                    backpack_coord = (W / 2, H / 2)
                else:
                    backpack_coord = (float(backpack_coord[0]), float(backpack_coord[1]))

                mask = create_heatmap_mask((H, W), backpack_coord, sigma=self.mask_sigma)
                mask = mask.unsqueeze(0)
                image = torch.cat([image, mask], dim=0)

            x, y, w, h = annotation['bbox']
            frame_id = extract_frame_id(file_name)

            return {
                'image': image,
                'label': label,
                'image_path': img_path,
                'bbox': torch.tensor([x, y, w, h], dtype=torch.float32),
                'frame_id': frame_id
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            # Return a default sample with zeros
            return {
                'image': torch.zeros(4 if self.use_mask else 3, 224, 224),
                'label': 0,
                'image_path': '',
                'bbox': torch.zeros(4, dtype=torch.float32),
                'frame_id': 'unknown_frame'
            }
