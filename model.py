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
        self.epoch_losses = []      # For plotting
        self.epoch_accuracies = []  # For plotting

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
            'predictions':[],
            'labels':[],
            'image_paths':[],
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
                # all_preds.extend(preds.cpu().numpy().astype(int).tolist())
                # all_labels.extend(labels.cpu().numpy().astype(int).tolist())

        if json_filename:
            # results = {
            # 'predictions': all_preds,
            # 'labels': all_labels,
            # 'image_path' : image_path, 
            # 'bbox': bbox,
            # }

            with open(json_filename, 'wb') as f:
                pickle.dump(results, f)
        
            print(f'Predictions saved to {json_filename}')

        loss_avg = total_loss/len(dataloader)
        accuracy = accuracy_score(results['predictions'],results['labels'] )
        print(f'Accuracy: {accuracy:.4f} Loss: {loss_avg:.4f}')
        return accuracy 
    
    def run_model(self, num_epoch, train_loader, eval_loader, view, scheduler=None):
        best_epochs = 0 # for early stopping
        for epoch in range(num_epoch):
            running_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", unit="batch")):
            #for batch in train_loader:
                loss = self.train(batch)
                running_loss += loss.item() * len(batch)

            epoch_loss = running_loss / len(train_loader)
            accuracy = self.evaluate(eval_loader)
            print(f'Epoch {epoch}, train loss: {epoch_loss}, eval accuracy: {accuracy}')

            # For plotting
            self.epoch_losses.append(epoch_loss)
            self.epoch_accuracies.append(accuracy)

            # update the scheduler
            if scheduler is not None:
                scheduler.step(accuracy)

            early_stoppage = 50 # epochs
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                best_epochs = 0 # reset count when accuracy improves
            else:
                best_epochs += 1
                if best_epochs >= early_stoppage:
                    print(f'No improvement over {early_stoppage} epochs, stopping early.')
                    break

            if (epoch+1)%5+1:
                save_checkpoint(self.model, self.optimizer, epoch, epoch_loss, accuracy, filename=view+'_ckpt.pth')
                self.best_accuracy = save_best_model(self.model, accuracy, self.best_accuracy, filename=view+'_best_model.pth')

        # Plot metrics after training
        epochs = list(range(len(self.epoch_losses)))
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.epoch_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.epoch_accuracies, label='Eval Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Accuracy over Epochs')
        plt.legend()
        plt.tight_layout()

        output_dir = '/mydata/vocim/zachary/color_prediction'
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(output_path)
        plt.close() 
    
    
    def load_model(self, ckpt):
        try:
            self.model, self.best_accuracy = load_checkpoint(self.model, ckpt)
            print(f'Loaded {ckpt} with accuracy {self.best_accuracy}')
            return self.best_accuracy
        except FileNotFoundError:
            return math.inf
