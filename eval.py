from PIL import Image
import torch
import torch.nn as nn
import timm 
from torch.optim import AdamW
from transformers import ResNetForImageClassification
from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder

def main(eval_json_data, img_dir = '/mydata/vocim/zachary/data/cropped'):
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model...")
    num_classes = 8
    model = timm.create_model('tiny_vit_21m_512.dist_in22k_ft_in1k', pretrained=False)
    tmp_in_features = model.head.in_features  
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  
        nn.Flatten(),             
        nn.Dropout(0.3),
        nn.Linear(tmp_in_features, num_classes)
    )
    model = model.to(device)
    print("Model created.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    #train_loader = get_train_dataloder(train_json_data, img_dir, batch_size=batch_size)
    eval_loader = get_eval_dataloder(eval_json_data, img_dir, batch_size=batch_size)
    print("Number of evaluation batches: ", len(eval_loader))

    trainer = Trainer(model = model, loss = criterion, optimizer = optimizer, device = device)
    loaded_acc = trainer.load_model(ckpt='top_colorid_best_model.pth')
    print(f"Loaded checkpoint with best accuracy: {loaded_acc}")

    #for debugging 
    for i, sample in enumerate(eval_loader):
        print(f"Batch {i} contains {sample['image'].size(0)} samples.")
        break

    print("Evaluating model...")
    trainer.evaluate(eval_loader, json_filename='output_top.pkl')
    print("Evaluation complete.")

if __name__=="__main__":
    eval_json_data='data/newdata_test_vidsplit_n.json'
    main(eval_json_data = eval_json_data)

