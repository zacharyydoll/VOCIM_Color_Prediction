from PIL import Image
import torch
import torch.nn as nn
import timm 
from torch.optim import AdamW
from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloader, get_train_dataloader
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
from color_gnn import extract_frame_id
import argparse


def main(eval_json_data=None, img_dir=None, model_path=None, output_dir=None):
    eval_batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # default args 
    if eval_json_data is None:
        eval_json_data = '/mydata/vocim/zachary/color_prediction/data/newdata_test_vidsplit_n.json'
    if img_dir is None:
        img_dir = '/mydata/vocim/zachary/data/cropped'
    if model_path is None:
        model_path = '/mydata/vocim/zachary/color_prediction/gnn_enhancement/embedding_bird_nodes_5_lay/top_colorid_best_model.pth'
    if output_dir is None:
        output_dir = 'eval_results'

    model = build_model(pretrained=True, dropout_rate=dropout_rate, num_classes=num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    eval_loader = get_eval_dataloader(eval_json_data, img_dir, batch_size=eval_batch_size, num_workers=0)

    trainer = Trainer(model=model, loss=criterion, optimizer=optimizer, device=device)
    loaded_acc = trainer.load_model(ckpt=model_path)

    metrics = evaluate_model(model, eval_loader, device, num_classes)
    
    # save metrics to output_dir
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, 'eval_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # import subprocess
    # subprocess.run(['python3', 'evaluation_metrics.py', '--output_dir', output_dir])
    
    return metrics

def evaluate_model(model, dataloader, device, num_classes, checkpoint_path=None):
    if checkpoint_path:
        model, _ = load_checkpoint(model, checkpoint_path)
    
    model.eval()
    evaluator = ModelEvaluator(num_classes)
    gnn_soft_outputs = []  # soft GNN outputs for analysis
    tinyvit_probabilities = []  # TinyViT softmax probabilities for analysis
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            image_paths = batch['image_path']  # get img paths from batch
            
            # TinyViT (pre-GNN) outputs
            tinyvit_logits = model(
                images,
                image_paths=image_paths,
                use_gnn=False
            )
            tinyvit_probs = torch.nn.functional.softmax(tinyvit_logits, dim=1)
            tinyvit_probabilities.extend(tinyvit_probs.cpu().numpy())

            # GNN-enhanced outputs (for inference)
            outputs = model(images)  # This is the hard-assignment output for inference
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            evaluator.update(outputs, labels, image_paths=image_paths)

            # soft GNN outputs for confidence analysis 
            if hasattr(model, 'color_gnn'):
                embeddings = model.forward_features(images)
                logits = model._original_forward(images)
                probs = torch.nn.functional.softmax(logits, dim=1)
                # group by frame
                from collections import defaultdict
                frames = defaultdict(list)
                emb_frames = defaultdict(list)
                for i, pth in enumerate(image_paths):
                    fid = extract_frame_id(pth)
                    frames[fid].append((i, probs[i]))
                    emb_frames[fid].append((i, embeddings[i]))
                for fid in frames:
                    lst = frames[fid]
                    emb_lst = emb_frames[fid]
                    idxs, ps = zip(*sorted(lst, key=lambda x: x[0]))
                    _, es = zip(*sorted(emb_lst, key=lambda x: x[0]))
                    P = torch.stack(ps)
                    E = torch.stack(es)
                    soft_out = model.color_gnn.forward_combined(E, P)
                    gnn_soft_outputs.extend(soft_out.cpu().numpy())
    
    metrics = evaluator.compute_metrics()
    metrics['gnn_soft_outputs'] = gnn_soft_outputs
    metrics['tinyvit_probabilities'] = tinyvit_probabilities
    
    return metrics

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='Path to test JSON file')
    parser.add_argument('--img_dir', type=str, default=None, help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation results')
    args = parser.parse_args()
    main(eval_json_data=args.data_path, img_dir=args.img_dir, model_path=args.model_path, output_dir=args.output_dir)