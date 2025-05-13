import os
import argparse

from PIL import Image
import torch
import timm 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder
from config import (
    use_glan, batch_size, num_epochs, dropout_rate, learning_rate, weight_decay, 
    scheduler_factor, scheduler_patience, num_classes, model_name, glan_early_stop, glan_weight_decay,
    smoothing, sigma_val, use_heatmap_mask, model_used, glan_dropout, glan_lr, glan_epochs, freeze_tinyvit
)
from model_builder import build_model
                   
def smooth_cross_entropy(pred, target, smoothing=smoothing):
    n_classes = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
    one_hot = one_hot * (1 - smoothing) + (smoothing / n_classes)
    log_prob = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prob).sum(dim=1).mean()
    return loss

def main(train_json_data, eval_json_data, img_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_epochs = glan_epochs if use_glan else num_epochs

    model = build_model(pretrained=True, dropout_rate=dropout_rate, num_classes=num_classes)
    model = model.to(device)

    #load TinyViT checkpoint first 
    pretrained_ckpt = "/mydata/vocim/zachary/color_prediction/TinyViT_with_mask/top_colorid_best_model_9831v_9765t.pth"
    if os.path.exists(pretrained_ckpt):
        print(f"Loading pretrained TinyViT from {pretrained_ckpt}")
        tmp = Trainer(model=model, loss=nn.NLLLoss(), optimizer=None, device=device)
        tmp.load_model(pretrained_ckpt)
        del tmp
    else:
        print(f"Warning: checkpoint not found at {pretrained_ckpt}, training from scratch")

    if use_glan: # Freeze all tinyViT params if using GLAN so only GNN trains 
        if freeze_tinyvit: 
            for name, p in model.named_parameters():
                p.requires_grad = name.startswith("color_gnn.")
            
            optimizer = optim.AdamW(
                model.color_gnn.parameters(),
                lr=glan_lr,
                weight_decay=glan_weight_decay
            )
        else: # unfreeze last block of tinyvit 
            for p in model.parameters():
                p.requires_grad = False

            # 2) unfreeze GNN
            gnn_params = list(model.color_gnn.parameters())
            for p in gnn_params:
                p.requires_grad = True

            # 3) unfreeze last *transformer* block
            last_stage   = model.stages[-1]
            last_block   = last_stage.blocks[-1]
            backbone_params = list(last_block.parameters())

            # 4) unfreeze the classification head
            head_params = list(model.head.parameters())

            for p in backbone_params + head_params:
                p.requires_grad = True

            # 5) optimizer with two param-groups
            optimizer = optim.AdamW(
                [
                { 'params': backbone_params + head_params,
                    'lr': learning_rate,
                    'weight_decay': weight_decay },
                { 'params': gnn_params,
                    'lr': glan_lr,
                    'weight_decay': glan_weight_decay }
                ]
            )
    else:
        # no GLAN: fine-tune everything
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    criterion = nn.NLLLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max', 
        factor=scheduler_factor, 
        patience=scheduler_patience, 
        verbose=True
    )  
    
    summary = f"""
    Training Summary:
    ----------------------------------------
    Device: {device}
    Using: {model_used}
    Model: {model.__class__.__name__}
    Pretrained: True
    Batch size: {batch_size}
    Number of epochs: {train_epochs}
    Learning Rate: {optimizer.param_groups[0]['lr']}
    Weight Decay: {optimizer.defaults.get('weight_decay', 'N/A')}
    Tiny ViT Completely Frozen: {freeze_tinyvit}

    Train JSON: {train_json_data}
    Eval JSON: {eval_json_data}
    Image Directory: {img_dir}

    Transformer Dropout Rate: {dropout_rate}

    Using GLAN: {use_glan}
    GLAN Dropout Rate: {glan_dropout}
    GLAN Epochs: {glan_epochs}
    GLAN Learning Rate: {glan_lr}
    GLAN Early Stoppage: After {glan_early_stop} epochs of no accuracy improvement
    GLAN weight decay: {glan_weight_decay}

    Smoothing: {smoothing}
    Use Heatmap Mask: {use_heatmap_mask}
    Mask Sigma: {sigma_val}
    Scheduler: ReduceLROnPlateau, Mode: max, Factor: {scheduler_factor}, Patience: {scheduler_patience}
    ----------------------------------------
    """
    print(summary)
    os.makedirs("logs", exist_ok=True)
    with open("logs/output_summary.log", "w") as f:
        f.write(summary)

    train_loader = get_train_dataloder(train_json_data, img_dir, batch_size=batch_size)
    eval_loader = get_eval_dataloder(eval_json_data, img_dir, batch_size=batch_size)
    
    trainer = Trainer(model=model, loss=criterion, optimizer=optimizer, device=device)
    
    # Load previous best model if it exists
    if os.path.exists('top_colorid_best_model.pth'):
        trainer.load_model('top_colorid_best_model.pth')
    
    # Run training
    trainer.run_model(num_epoch=train_epochs, train_loader=train_loader, 
                     eval_loader=eval_loader, view='top_colorid', scheduler=scheduler)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_data', type=str)
    parser.add_argument('--eval_json_data', type=str)
    parser.add_argument('--img_dir', type=str)
    args = parser.parse_args()

    main(args.train_json_data, args.eval_json_data, args.img_dir)