from torch.utils.data import DataLoader
from dataset import ImageDataset
from sampler import ClassBalancedSampler
from PIL import Image
from PIL import ImageOps
from torchvision import transforms
from config import sampler_ambig_factor

def letterbox(img, size=(512, 512), fill_color=(0, 0, 0)):
    """
    Resize image to fit within the size, while preserving aspect ratio, then pad 
    with specified color such that the final output is exactly size (512 for tiny_vit_21m_512)
    """
    # ENSURING WE RETURN A PIL IMG!!
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return ImageOps.pad(img, size, method=Image.BICUBIC, color=fill_color)

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: letterbox(img, size=(512, 512))),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10), # rotate images by +/- 10 degrees (2)
    transforms.RandomAffine(
         degrees=10,              # further rotation 
         translate=(0.1, 0.1),    # random translation in x and y directions (10% of image size)
         scale=(0.9, 1.1),        # random scaling between 90% and 110%
         shear=5                  # random shear in degrees
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def get_train_dataloder(json_file, img_dir, batch_size, ambiguous_json_path=None, ambiguous_factor=sampler_ambig_factor):
    transform = train_transform # using letteboxing to 512
    train_data = ImageDataset(data_path = json_file, img_dir = img_dir, transform=transform)
    sampler = ClassBalancedSampler(train_data, ambiguous_json_path=ambiguous_json_path, ambiguous_factor=ambiguous_factor).get_sampler()
    train_dataloader = DataLoader(train_data, sampler = sampler, batch_size=batch_size)
    return train_dataloader

def get_eval_dataloder(json_file, img_dir, batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: letterbox(img, size=(512, 512))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_data = ImageDataset(data_path = json_file, img_dir = img_dir, transform=transform)
    # sampler = ClassBalancedSampler(eval_data.labels).get_sampler()
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return eval_dataloader