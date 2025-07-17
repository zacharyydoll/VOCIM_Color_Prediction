import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import yaml  
from config import sigma_val, use_heatmap_mask
import torchvision.transforms as transforms
from dataloader import letterbox

def transform_backpack_coords(original_coords, original_size, target_size=(512, 512)):
    """
    Transform backpack coordinates from original image space to letterboxed space.
    
    Args:
        original_coords: (x, y) in original image coordinates
        original_size: (width, height) of original image
        target_size: (width, height) of target image (512, 512)
    
    Returns:
        (x, y) in target image coordinates
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    # Calculate scaling factors for letterboxing
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    scale = min(scale_x, scale_y)  # Letterboxing uses the smaller scale
    
    # Scale coordinates
    scaled_x = original_coords[0] * scale
    scaled_y = original_coords[1] * scale
    
    # Calculate padding offsets
    new_w = orig_w * scale
    new_h = orig_h * scale
    offset_x = (target_w - new_w) / 2
    offset_y = (target_h - new_h) / 2
    
    # Apply offsets
    final_x = scaled_x + offset_x
    final_y = scaled_y + offset_y
    
    return (final_x, final_y)

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
            
            # Create heatmap mask BEFORE any transforms (using original image coordinates)
            if self.use_mask: 
                img_info = self.img_paths[img_idx]
                orig_w, orig_h = img_info['width'], img_info['height']
                backpack_coord = img_info.get("backpack_coord", None)
                if backpack_coord is None:
                    backpack_coord = (orig_w / 2, orig_h / 2)
                else:
                    backpack_coord = (float(backpack_coord[0]), float(backpack_coord[1]))

                # Create mask using original image coordinates
                mask = create_heatmap_mask((orig_h, orig_w), backpack_coord, sigma=self.mask_sigma)
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
                
                # Convert PIL image to tensor and add mask channel
                image_tensor = transforms.ToTensor()(image)  # (3, H, W)
                image_with_mask = torch.cat([image_tensor, mask], dim=0)  # (4, H, W)
                
                # Apply the same geometric transforms to both RGB and mask
                rgb_tensor = image_with_mask[:3]
                mask_tensor = image_with_mask[3:]
                
                # Convert to PIL for transforms
                rgb_pil = transforms.ToPILImage()(rgb_tensor)
                mask_pil = transforms.ToPILImage()(mask_tensor)
                
                # Apply letterboxing to both
                rgb_letterboxed = letterbox(rgb_pil, size=(512, 512))
                mask_letterboxed = letterbox(mask_pil, size=(512, 512))
                
                # Convert back to tensors
                rgb_tensor = transforms.ToTensor()(rgb_letterboxed)
                mask_tensor = transforms.ToTensor()(mask_letterboxed)
                
                # Apply normalization only to RGB
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                rgb_tensor = (rgb_tensor - mean) / std
                
                # Combine RGB and mask
                image = torch.cat([rgb_tensor, mask_tensor], dim=0)
            else:
                # Apply transforms to image
                if self.transform:
                    image = self.transform(image)  # expected shape: (3, H, W)

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
            # Default sample with zeros
            return {
                'image': torch.zeros(4 if self.use_mask else 3, 224, 224),
                'label': 0,
                'image_path': '',
                'bbox': torch.zeros(4, dtype=torch.float32),
                'frame_id': 'unknown_frame'
            }
