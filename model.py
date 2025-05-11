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
from config import use_glan, compute_graph_metrics, num_classes
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
        self.epoch_accuracies = []      # Evaluation accuracy per epoch
        self.epoch_eval_losses = []     # Evaluation loss per epoch
        if use_glan:
            self.evaluator = ModelEvaluator(num_classes=num_classes)

    def train(self, batch):
        self.model.train()
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        image_paths = batch['image_path']
        
        # Forward pass without GNN during training
        outputs = self.model(images, image_paths=image_paths, use_gnn=False)
        loss = self.loss(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, batch):
        self.model.eval()
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        image_paths = batch['image_path']
        
        with torch.no_grad():
            # Use GNN during evaluation
            outputs = self.model(images, image_paths=image_paths, use_gnn=True)
            loss = self.loss(outputs, labels)
            
        return loss.item(), outputs

    def run_model(self, num_epoch, train_loader, eval_loader, view, scheduler=None):
        best_epochs = 0  # for early stopping
        for epoch in range(num_epoch):
            epoch_start = time.time()
            running_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", unit="batch")):
                loss = self.train(batch)
                running_loss += loss * len(batch)

            epoch_train_loss = running_loss / len(train_loader)

            eval_accuracy, eval_loss = self.evaluate(eval_loader)
            log_message(f"\nEpoch {epoch}, Train Loss: {epoch_train_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}, Eval Loss: {eval_loss:.4f}")

            # Store losses and accuracy
            self.epoch_losses.append(epoch_train_loss)
            self.epoch_eval_losses.append(eval_loss)
            self.epoch_accuracies.append(eval_accuracy)

            # Update the scheduler
            if scheduler is not None:
                scheduler.step(eval_accuracy)

            early_stoppage = 10  # epochs
            if eval_accuracy > self.best_accuracy:
                self.best_accuracy = save_best_model(self.model, eval_accuracy, self.best_accuracy, filename=view+'_best_model.pth')
                log_message(f"New best model with {self.best_accuracy:.4f} accuracy saved to {view+'_best_model.pth'}")
                best_epochs = 0  # reset count when accuracy improves
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
                save_checkpoint(self.model, self.optimizer, epoch, epoch_train_loss, eval_accuracy, filename=view+'_ckpt.pth')
                log_message(f"Checkpoint saved to {view+'_colorid_ckpt.pth'}")
        log_message("\nTraining complete.")


    def load_model(self, ckpt):
        try:
            self.model, self.best_accuracy = load_checkpoint(self.model, ckpt)
            print(f'Loaded {ckpt} with accuracy {self.best_accuracy}')
            return self.best_accuracy
        except FileNotFoundError:
            return math.inf
