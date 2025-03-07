from torch.utils.data import DataLoader
from dataset import ImageDataset
from sampler import ClassBalancedSampler
from PIL import ImageOps
from torchvision import transforms

def letterbox(img, size=(512, 512), fill_color=(0, 0, 0)):
    """
    Resize the image to fit within the size, while preserving aspect ratio, then pad 
    with specified color such that the final output is exactly `size`(512 for tiny_vit_21m_512)
    """
    return ImageOps.pad(img, size, method=Image.BICUBIC, color=fill_color)

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: letterbox(img, size=(512, 512))),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def get_train_dataloder(json_file, img_dir, batch_size):
    transform = train_transform # using letteboxing to 512
    train_data = ImageDataset(data_path = json_file, img_dir = img_dir, transform=transform)
    sampler = ClassBalancedSampler(train_data.annotations).get_sampler()
    train_dataloader = DataLoader(train_data, sampler = sampler, batch_size=batch_size)
    return train_dataloader

def get_eval_dataloder(json_file, img_dir, batch_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: letterbox(img, size=(512, 512))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_data = ImageDataset(data_path = json_file, img_dir = img_dir, transform=transform)
    # sampler = ClassBalancedSampler(eval_data.labels).get_sampler()
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    return eval_dataloader