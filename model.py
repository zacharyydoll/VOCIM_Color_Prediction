import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import math
import collections
import os
import json
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score
from utils import save_checkpoint, save_best_model, load_checkpoint
import pdb

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

    def train(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images = batch['image']
        labels = batch['label']
        images, labels = images.to(self.device), labels.to(self.device)

        outputs = self.model(images)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, dataloader, json_filename=None):
        self.model.eval()
        total_loss = 0
        results = {
            'predictions': [],
            'labels': [],
            'image_paths': [],
            'bboxes': [],
        }

        with torch.no_grad():
            for sample in dataloader:
                images = sample['image']
                labels = sample['label']
                image_path = sample['image_path']
                bbox = sample['bbox'][0]

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                total_loss += loss * len(sample)
                _, preds = torch.max(outputs, 1)

                results['predictions'].extend(preds.cpu().numpy().astype(int).tolist())
                results['labels'].extend(labels.cpu().numpy().astype(int).tolist())
                results['image_paths'].extend(image_path)
                results['bboxes'].extend(bbox.cpu().numpy().tolist())

        loss_avg = total_loss / len(dataloader)
        accuracy = accuracy_score(results['predictions'], results['labels'])
        print(f'Accuracy: {accuracy:.4f} Loss: {loss_avg:.4f}')

        if json_filename:
            with open(json_filename, 'wb') as f:
                pickle.dump(results, f)
            print(f'Predictions saved to {json_filename}')
        
        return accuracy, loss_avg  # Return both evaluation accuracy and loss
    
    def run_model(self, num_epoch, train_loader, eval_loader, view, scheduler=None):
        best_epochs = 0  # for early stopping
        for epoch in range(num_epoch):
            running_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", unit="batch")):
                loss = self.train(batch)
                running_loss += loss.item() * len(batch)

            epoch_train_loss = running_loss / len(train_loader)
            # Evaluate returns (accuracy, eval_loss)
            eval_accuracy, eval_loss = self.evaluate(eval_loader)
            print(f'Epoch {epoch}, Train Loss: {epoch_train_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}, Eval Loss: {eval_loss:.4f}')

            # Store losses and accuracy
            self.epoch_losses.append(epoch_train_loss)
            self.epoch_eval_losses.append(eval_loss)
            self.epoch_accuracies.append(eval_accuracy)

            # Update the scheduler
            if scheduler is not None:
                scheduler.step(eval_accuracy)

            early_stoppage = 12  # epochs
            if eval_accuracy > self.best_accuracy:
                self.best_accuracy = save_best_model(self.model, eval_accuracy, self.best_accuracy, filename=view+'_best_model.pth')
                best_epochs = 0  # reset count when accuracy improves
            else:
                best_epochs += 1
                if best_epochs >= early_stoppage:
                    print(f'No improvement over {early_stoppage} epochs, stopping early.')
                    break

            epoch_log = f"No Improvement count: {best_epochs} "
            print(epoch_log)
            with open("logs/output_summary.log", "a") as f:
                f.write(epoch_log)

            if (epoch+1) % 5 + 1:
                save_checkpoint(self.model, self.optimizer, epoch, epoch_train_loss, eval_accuracy, filename=view+'_ckpt.pth')

        # Plot training and evaluation loss on the same graph
        epochs = list(range(len(self.epoch_losses)))
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.epoch_losses, label='Training Loss')
        plt.plot(epochs, self.epoch_eval_losses, label='Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss over Epochs')
        plt.legend()
        plt.tight_layout()

        output_dir = '/mydata/vocim/zachary/color_prediction'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(output_path)
        print(f'Loss curve saved to {output_path}')
        plt.close()

        # Also plot evaluation accuracy separately if desired
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.epoch_accuracies, label='Evaluation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Accuracy over Epochs')
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'accuracy_curve.png')
        plt.savefig(output_path)
        print(f'Accuracy curve saved to {output_path}')
        plt.close()

    def load_model(self, ckpt):
        try:
            self.model, self.best_accuracy = load_checkpoint(self.model, ckpt)
            print(f'Loaded {ckpt} with accuracy {self.best_accuracy}')
            return self.best_accuracy
        except FileNotFoundError:
            return math.inf
