import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import re

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
        label = extract_identity_number(identity_str)

        img_idx = annotation['image_id']
        img_path = os.path.join(self.img_dir, self.img_paths[img_idx]['file_name'])
        assert os.path.exists(img_path), f"image doesn't exist at {img_path}"

        image = Image.open(img_path)
        x, y, w, h = annotation['bbox']
        cropped_image = image.crop((x, y, x + w, y + h))
        
        if self.transform:
            cropped_image = self.transform(cropped_image)

        sample = {
            'image': cropped_image,
            'label': label,
            'image_path': img_path,
            'bbox': [torch.tensor([x, y, w, h])]
        }
        return sample
