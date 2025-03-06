from torch.utils.data import DataLoader
from dataset import ImageDataset
from sampler import ClassBalancedSampler
from torchvision import transforms

def get_train_dataloder(json_file, img_dir, batch_size):
    transform = transforms.Compose([
    transforms.Resize((224,224)),  # Randomly crop and resize the image
    #transforms.RandomRotation(30),      # Randomly rotate the image by up to 30 degrees
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
    transforms.ToTensor(),              # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    train_data = ImageDataset(data_path = json_file, img_dir = img_dir, transform=transform)
    sampler = ClassBalancedSampler(train_data.annotations).get_sampler()
    train_dataloader = DataLoader(train_data, sampler = sampler, batch_size=batch_size)
    return train_dataloader

def get_eval_dataloder(json_file, img_dir, batch_size):
    transform = transforms.Compose([
    transforms.Resize((224,224)),  # Randomly crop and resize the image
    #transforms.RandomRotation(30),      # Randomly rotate the image by up to 30 degrees
    #transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
    transforms.ToTensor(),              # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    eval_data = ImageDataset(data_path = json_file, img_dir = img_dir, transform=transform)
    # sampler = ClassBalancedSampler(eval_data.labels).get_sampler()
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    return eval_dataloader