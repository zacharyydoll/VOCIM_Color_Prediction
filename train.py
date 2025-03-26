import os
import argparse

from PIL import Image
import torch
import timm 
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
# from transformers import ResNetForImageClassification

from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import batch_size, num_epochs, dropout_rate, learning_rate, weight_decay, scheduler_factor, scheduler_patience, num_classes, model_name, smoothing, sigma_val, use_heatmap_mask, model_used
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

    # model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
    # model.classifier = torch.nn.Sequential(
        # torch.nn.Flatten(start_dim=1, end_dim=-1),
        # torch.nn.Linear(in_features=2048, out_features=8, bias=True))

    model = build_model(pretrained=True, dropout_rate=dropout_rate, num_classes=num_classes)
    model = model.to(device)

    # freezing early layers to reduce overfitting 
    #for name, param in model.named_parameters():
    #    if "patch_embed" in name:
    #        param.requires_grad = False

    #class_counts = [2386, 2247, 825, 801, 358, 300, 315, 281] # check with utils_and_analysis/class_distribution_all.py
    #weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    #weights = weights / weights.sum() * len(weights)  # normalize so average weight is approx 1
    #weights = weights.to(device)

    criterion = nn.CrossEntropyLoss()
    #criterion = smooth_cross_entropy
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True) 

    summary = f"""
    Training Summary:
    ---------------------
    Using: {model_used}
    Model: {model.__class__.__name__}
    Pretrained: True
    Batch size: {batch_size}
    Number of epochs: {num_epochs}
    Learning Rate: {optimizer.param_groups[0]['lr']}
    Weight Decay: {optimizer.defaults.get('weight_decay', 'N/A')}
    Dropout Rate: {dropout_rate}
    Device: {device}
    Train JSON: {train_json_data}
    Eval JSON: {eval_json_data}
    Image Directory: {img_dir}
    Smoothing: {smoothing}
    Use Heatmap Mask: {use_heatmap_mask}
    Mask Sigma: {sigma_val}
    Scheduler: ReduceLROnPlateau, Mode: min, Factor: {scheduler_factor}, Patience: {scheduler_patience}
    ---------------------
    """
    print(summary)
    os.makedirs("logs", exist_ok=True)
    with open("logs/output_summary.log", "w") as f: # write to summary as well
        f.write(summary)

    train_loader = get_train_dataloder(train_json_data, img_dir, batch_size=batch_size)
    eval_loader = get_eval_dataloder(eval_json_data, img_dir, batch_size=batch_size)
    
    trainer = Trainer(model=model, loss=criterion, optimizer=optimizer, device=device)
    if os.path.exists('top_colorid_best_model.pth'):
        trainer.load_model('top_colorid_best_model.pth')
    trainer.run_model(num_epoch = num_epochs, train_loader=train_loader, eval_loader=eval_loader, view = 'top_colorid', scheduler= scheduler)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_data', type=str)
    parser.add_argument('--eval_json_data', type=str)
    parser.add_argument('--img_dir', type=str)
    args = parser.parse_args()

    main(args.train_json_data, args.eval_json_data, args.img_dir)