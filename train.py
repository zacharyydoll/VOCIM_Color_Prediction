import os
import argparse

from PIL import Image
import torch
import timm 
import torch.nn as nn

from torch.optim import AdamW
# from transformers import ResNetForImageClassification

from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder

def main(train_json_data, eval_json_data, img_dir):
    # train_json_data = 'data/'+view+'_rtmdet_train_colorid_vocim.json'
    # eval_json_data = 'data/'+view+'_rtmdet_val_colorid_vocim.json'
    # if view=='top':
    #     img_dir = '/mydata/vocim/xiaoran/scripts/mmpose/data/vocim/images'
    # else:
    #     img_dir = '/mydata/vocim/xiaoran/scripts/mmpose/data/vocim/images_'+view
    # batch_size = 32
    #train_json_data = 'data/'+view+'_rtmdet_train_colorid_vocim.json'
    #eval_json_data = 'data/'+view+'_rtmdet_val_colorid_vocim.json'
    #if view=='top':
    #    img_dir = '/mydata/vocim/xiaoran/scripts/mmpose/data/vocim/images'
    #else:
    #    img_dir = '/mydata/vocim/xiaoran/scripts/mmpose/data/vocim/images_'+view
    batch_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
    # model.classifier = torch.nn.Sequential(
        # torch.nn.Flatten(start_dim=1, end_dim=-1),
        # torch.nn.Linear(in_features=2048, out_features=8, bias=True))
    model = timm.create_model('tiny_vit_21m_512.dist_in22k_ft_in1k', pretrained=True)
    #print("Head in_features:", model.head.in_features)
    in_features = model.head.in_features
    num_classes = 8
    #model.head = nn.Linear(in_features, num_classes)
    #model.head = nn.Linear(model.head.in_features, num_classes)
    # Add global average pooling
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # Reduces spatial dimensions to (batch_size, 576, 1, 1)
        nn.Flatten(),             # Flattens to (batch_size, 576)
        nn.Dropout(0.3),          # added dropout of 0.3, may need to tweak
        nn.Linear(model.head.in_features, num_classes)  # Final classification layer
    )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    train_loader = get_train_dataloder(train_json_data, img_dir, batch_size=batch_size)

    eval_loader = get_eval_dataloder(eval_json_data, img_dir, batch_size=batch_size)
    
    # img_dir relative path should be "/mydata/vocim/zachary/data/cropped"
    trainer = Trainer(model = model, loss = criterion, optimizer = optimizer, device = device)
    if os.path.exists('top_colorid_best_model.pth'):
        trainer.load_model('top_colorid_best_model.pth')
    trainer.run_model(num_epoch = 50, train_loader=train_loader, eval_loader=eval_loader, view = 'top_colorid')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_data', type=str)
    parser.add_argument('--eval_json_data', type=str)
    parser.add_argument('--img_dir', type=str)
    args = parser.parse_args()

    main(args.train_json_data, args.eval_json_data, args.img_dir)