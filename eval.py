from PIL import Image
import torch
import torch.nn as nn
import timm 
from torch.optim import AdamW
from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder
from config import (
    batch_size, num_epochs, dropout_rate, 
    learning_rate, weight_decay, scheduler_factor, 
    scheduler_patience, num_classes, model_name
)
from model_builder import build_model
from tqdm import tqdm
import pickle
from evaluation_metrics import ModelEvaluator
from utils import load_checkpoint
import os


def main(eval_json_data, img_dir = '/mydata/vocim/zachary/data/cropped'):
    eval_batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=True, dropout_rate=dropout_rate, num_classes=num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    eval_loader = get_eval_dataloder(eval_json_data, img_dir, batch_size=eval_batch_size, num_workers=0)

    trainer = Trainer(model=model, loss=criterion, optimizer=optimizer, device=device)
    loaded_acc = trainer.load_model(ckpt='/mydata/vocim/zachary/color_prediction/gnn_enhancement/top_colorid_best_model.pth')

    metrics = evaluate_model(model, eval_loader, device, num_classes)
    
    # Save metrics to eval_results directory
    os.makedirs('eval_results', exist_ok=True)
    pkl_path = os.path.join('eval_results', 'evaluation_metrics.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Call evaluation_metrics.py to generate the .log file and images
    import subprocess
    subprocess.run(['python3', 'evaluation_metrics.py'])
    
    return metrics

def evaluate_model(model, dataloader, device, num_classes, checkpoint_path=None):
    if checkpoint_path:
        model, _ = load_checkpoint(model, checkpoint_path)
    
    model.eval()
    evaluator = ModelEvaluator(num_classes)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            image_paths = batch['image_path']  # Get image paths from batch
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            evaluator.update(outputs, labels, image_paths=image_paths)
    
    metrics = evaluator.compute_metrics()
    
    return metrics

if __name__=="__main__":
    eval_json_data='/mydata/vocim/zachary/color_prediction/data/mult_bkpk_sub_test_set.json'
    main(eval_json_data = eval_json_data)