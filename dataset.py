import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import re

def create_heatmap_mask(image_size, center, sigma=5):
    """
    Creates a Gaussian heatmap mask of shape (H, W) where the maximum (1.0)
    is at the given center.
    image_size: tuple (H, W)
    center: tuple (x, y) coordinates in pixel space (x along width, y along height)
    sigma: controls the spread of the Gaussian
    """
    H, W = image_size
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    heatmap = np.exp(-((xv - center[0]) ** 2 + (yv - center[1]) ** 2) / (2 * sigma ** 2))
    return torch.tensor(heatmap, dtype=torch.float32)


def extract_identity_number(identity_str):
    """
    Extract the first integer from an identity string.
    e.g. "bird_y_1" returns 1. (see BP10_01 annotations)
    """
    if identity_str is None:
        raise ValueError("Received a None identity")
    match = re.search(r'\d+', identity_str)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No integer found in identity string: {identity_str}")

class ImageDataset(Dataset):
    def __init__(self, data_path, img_dir, transform=None):
        """
        Args:
            data_path (str): Path to the JSON annotations file.
            img_dir (str): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(data_path, 'r') as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.img_paths = data['images']
        self.annotations = data['annotations']
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        # Extract identity as string, then convert it to an int
        identity_str = annotation['identity']
        label = extract_identity_number(identity_str) - 1

        img_idx = annotation['image_id']
        img_path = os.path.join(self.img_dir, self.img_paths[img_idx]['file_name'])
        assert os.path.exists(img_path), f"image doesn't exist at {img_path}"

        # Fixed: Use the entire pre-cropped image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # expected shape (3, H, W)

        _, H, W = image.shape

        # Retrieve backpack coordinate from the corresponding image entry (not from annotation)
        img_info = self.img_paths[img_idx]
        backpack_coord = img_info.get("backpack_coord", None)
        if backpack_coord is None:
            backpack_coord = (W / 2, H / 2)
        else:
            # Use as (x, y) directly (do not swap) because your CSV provides (x, y)
            backpack_coord = (float(backpack_coord[0]), float(backpack_coord[1]))

        # Create the heatmap mask using the provided backpack coordinate.
        mask = create_heatmap_mask((H, W), backpack_coord, sigma=5)
        mask = mask.unsqueeze(0)
        
        # Concatenate the mask with the image to form a 4-channel input.
        image_with_mask = torch.cat([image, mask], dim=0)  # now shape: (4, H, W)

        # You can still use the original bounding box information if needed.
        x, y, w, h = annotation['bbox']

        sample = {
            'image': image_with_mask,  # 4-channel image (RGB + mask)
            'label': label,
            'image_path': img_path,
            'bbox': [torch.tensor([x, y, w, h])]
        }
        return sample

