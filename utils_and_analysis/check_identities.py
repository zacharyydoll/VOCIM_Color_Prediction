import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import ImageDataset

json_path = "/mydata/vocim/zachary/color_prediction/data/newdata_cls_train_vidsplit_n.json"
img_dir = "/mydata/vocim/zachary/data/cropped"
output_dir = "/mydata/vocim/zachary/color_prediction/utils_and_analysis/identity_check"
os.makedirs(output_dir, exist_ok=True)

# create the dataset
simple_transform = transforms.Compose([
    transforms.Lambda(lambda img: img),  # identity transform
    transforms.ToTensor()
])
dataset = ImageDataset(data_path=json_path, img_dir=img_dir, transform=simple_transform, mask_sigma=13)

import yaml
with open("/mydata/vocim/zachary/color_prediction/newdata_colormap.yaml", "r") as f:
    colormap = yaml.safe_load(f)
inv_colormap = {v: k for k, v in colormap.items()}

required_indices = [2, 4]
num_samples = 15 # adjust if need more sample displayed
if num_samples < len(required_indices):
    raise ValueError("num_samples must be at least as many as the required indices")
available_indices = set(range(len(dataset))) - set(required_indices)
random_indices = random.sample(available_indices, num_samples - len(required_indices))
indices = required_indices + random_indices

to_pil = transforms.ToPILImage()

for idx in indices:
    sample = dataset[idx]
    rgb_tensor = sample["image"][:3]
    rgb_image = to_pil(rgb_tensor)
    
    label = sample["label"]  # should be an int from 0 to 7
    # Lookup the corresponding color name in mapping
    color_name = inv_colormap.get(label, "unknown")
    
    # Annotate the image
    draw = ImageDraw.Draw(rgb_image)
    text = f"class: {label} ({color_name})"
    draw.text((15, 15), text, fill=(255, 255, 255))
    
    output_path = os.path.join(output_dir, f"sample_{idx}.png")
    rgb_image.save(output_path)
    print(f"Saved sample {idx} with label: {text}")