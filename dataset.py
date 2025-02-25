import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, data_path, img_dir, transform=None):
        """
        Args:
            image_paths (list of str): List of file paths to the images.
            labels (list of int): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(data_path, 'r') as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.img_paths = data['images']
        self.labels = data['annotations']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]['identity']
        img_idx = self.labels[idx]['image_id']
        img_path = os.path.join(self.img_dir, self.img_paths[img_idx]['file_name'])
        assert os.path.exists(img_path),f"image doesn't exist at {img_path}"

        image = Image.open(img_path) # Load and convert image to RGB
        x, y, w, h = self.labels[idx]['bbox']
        
        cropped_image = image.crop((x, y, x+w, y+h))
        
        if self.transform:
           cropped_image = self.transform(cropped_image)

        sample = {}
        sample['image'] = cropped_image
        sample['label'] = label
        sample['image_path'] = img_path
        sample['bbox'] = [torch.tensor([x, y, w, h])]
        return sample
