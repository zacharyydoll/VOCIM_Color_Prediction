import os
from dataset import ImageDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

def main():
    # Path to your merged JSON file and the base directory for cropped images
    data_path = '/mydata/vocim/zachary/data/cropped_merged_annotations.json'
    img_dir = '/mydata/vocim/zachary/data/cropped'
    
    # Define a simple transform (resize to 224x224 and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create the dataset instance
    dataset = ImageDataset(data_path=data_path, img_dir=img_dir, transform=transform)
    print("Number of samples in dataset:", len(dataset))
    
    # Get the first sample
    sample = dataset[0]
    print("Image path:", sample['image_path'])
    
    # If the image is a PIL Image, print its size; otherwise, if it's a tensor, convert and print shape
    image = sample['image']
    if isinstance(image, torch.Tensor):
        # Convert from C x H x W to H x W x C for display
        image_np = image.permute(1, 2, 0).numpy()
        print("Image tensor shape:", image_np.shape)
    else:
        print("Image size:", image.size)
    
    # Optionally, display the image using matplotlib
    plt.imshow(image if not torch.is_tensor(image) else image_np)
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
