import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import math
import collections
import os
import json
import time
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score
from utils import save_checkpoint, save_best_model, load_checkpoint
from evaluation_metrics import ModelEvaluator   
from config import use_glan, compute_graph_metrics, num_classes, glan_early_stop
import torch.nn.functional as F
import pdb

def log_message(message, log_file="logs/output_summary.log"):
    print(message)
    # Write to both the specified log file and the summary log
    for log_path in [log_file, "logs/output_summary.log"]:
        with open(log_path, "a") as f:
            f.write(message + "\n")

# Function to classify an image
class Trainer:
    def __init__(self, model, optimizer, loss, device):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss.to(device)
        self.device = device
        self.best_accuracy = 0
        self.epoch_losses = []        # Training loss per epoch
        self.epoch_accuracies = []    # Evaluation accuracy per epoch (GNN)
        self.epoch_eval_losses = []   # Evaluation loss per epoch
        if use_glan:
            self.evaluator = ModelEvaluator(num_classes=num_classes)

    def train(self, batch):
        self.model.train()
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        image_paths = batch['image_path']
        
        # Forward pass with GNN training
        outputs = self.model(
            images,
            image_paths=image_paths,
            use_gnn=True,
            train_gnn=True
        )

        loss = self.loss(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, eval_loader):
        self.model.eval()
        total_raw_correct = 0
        total_gnn_correct = 0
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                image_paths = batch['image_path']

                # Raw TinyViT outputs, compute log‐probs before NLLLoss
                raw_logits = self.model(
                    images,
                    image_paths=image_paths,
                    use_gnn=False
                )
                raw_logprobs = F.log_softmax(raw_logits, dim=1)
                raw_pred = raw_logits.argmax(dim=1)
                total_raw_correct += (raw_pred == labels).sum().item()

                # Loss on raw branch
                loss = self.loss(raw_logprobs, labels)
                total_loss += loss.item() * labels.size(0)

                # GNN‐enhanced outputs (already log‐softmax in train branch)
                gnn_logits = self.model(
                    images,
                    image_paths=image_paths,
                    use_gnn=True
                )
                gnn_pred = gnn_logits.argmax(dim=1)
                total_gnn_correct += (gnn_pred == labels).sum().item()

                total_samples += labels.size(0)

        raw_acc = total_raw_correct / total_samples
        gnn_acc = total_gnn_correct / total_samples
        avg_loss = total_loss / total_samples

        return raw_acc, gnn_acc, avg_loss


    def run_model(self, num_epoch, train_loader, eval_loader, view, scheduler=None):
        best_epochs = 0  # for early stopping
        for epoch in range(num_epoch):
            epoch_start = time.time()
            running_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", unit="batch")):
                loss = self.train(batch)
                running_loss += loss * len(batch)

            epoch_train_loss = running_loss / len(train_loader)

            # Evaluate both raw and GNN accuracies
            raw_acc, gnn_acc, eval_loss = self.evaluate(eval_loader)
            log_message(
                f"\nEpoch {epoch} — "
                f"Train Loss: {epoch_train_loss:.4f}  |  "
                f"Raw Acc: {raw_acc:.4f}  |  "
                f"GNN Acc: {gnn_acc:.4f}  |  "
                f"Eval Loss: {eval_loss:.4f}"
            )

            # Store metrics
            self.epoch_losses.append(epoch_train_loss)
            self.epoch_eval_losses.append(eval_loss)
            # track GNN accuracy for checkpointing/scheduling
            self.epoch_accuracies.append(gnn_acc)

            # Update the scheduler on GNN accuracy
            if scheduler is not None:
                scheduler.step(gnn_acc)

            # Early stopping & best model saving by GNN accuracy
            early_stoppage = glan_early_stop if use_glan else 10  
            if gnn_acc > self.best_accuracy:
                self.best_accuracy = save_best_model(
                    self.model, gnn_acc, self.best_accuracy,
                    filename=view + '_best_model.pth'
                )
                log_message(f"New best model with {self.best_accuracy:.4f} accuracy saved to {view+'_best_model.pth'}")
                best_epochs = 0
            else:
                best_epochs += 1
                if best_epochs >= early_stoppage:
                    log_message(f"\nNo improvement over {early_stoppage} epochs, stopping early with best model at {self.best_accuracy:.4f} accuracy.")
                    break

            log_message(f"No Improvement count: {best_epochs}")

            epoch_time = time.time() - epoch_start
            hours = int(epoch_time // 3600)
            minutes = int((epoch_time % 3600) // 60)
            seconds = int(epoch_time % 60)
            log_message(f"Epoch {epoch} took {hours}h {minutes}m {seconds}s")

            if (epoch+1) % 2 == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    epoch_train_loss, gnn_acc,
                    filename=view + '_ckpt.pth'
                )
                log_message(f"Checkpoint saved to {view+'_colorid_ckpt.pth'}")
        log_message("\nTraining complete.")

    def load_model(self, ckpt):
        try:
            self.model, self.best_accuracy = load_checkpoint(self.model, ckpt)
            print(f'Loaded {ckpt} with accuracy {self.best_accuracy}')
            return self.best_accuracy
        except FileNotFoundError:
            return math.inf
