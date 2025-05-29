import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
import random
from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder
from model_builder import build_model
from config import (
    batch_size, num_epochs, dropout_rate, learning_rate, weight_decay,
    scheduler_mode, scheduler_factor, scheduler_patience, num_classes, model_name,
    glan_early_stop, glan_weight_decay, smoothing, sigma_val, use_heatmap_mask, glan_hidden_dim,
    model_used, glan_dropout, glan_lr, glan_epochs, freeze_tinyvit, glan_num_layers, weigh_ambig_samples,
    sampler_ambig_factor
)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from tqdm import tqdm
from utils import save_checkpoint, save_best_model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_epochs = glan_epochs if args.use_glan else num_epochs

    # Build model with all layers trainable (not frozen)
    model = build_model(pretrained=True, dropout_rate=dropout_rate, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=scheduler_factor, patience=scheduler_patience)

    ambiguous_json_path = "data/ambig_train_samples.json"
    train_loader = get_train_dataloder(
        args.train_json_data,
        args.img_dir,
        batch_size=batch_size,
        ambiguous_json_path=ambiguous_json_path,
        ambiguous_factor=sampler_ambig_factor
    )
    eval_loader = get_eval_dataloder(args.eval_json_data, args.img_dir, batch_size=batch_size)

    logs_dir = os.path.abspath(os.path.join(args.output_dir, '..', 'logs'))
    os.makedirs(logs_dir, exist_ok=True)
    output_log_file = os.path.join(logs_dir, "output.log")
    summary_log_file = os.path.join(logs_dir, "output_summary.log")

    sys.stdout = open(output_log_file, 'w')
    sys.stderr = sys.stdout
    summary_fh = open(summary_log_file, 'w')

    def log_summary(message):
        print(message)
        summary_fh.write(message + "\n")
        summary_fh.flush()

    summary = f"""
    Training Summary:
    ---------------------------------------------------------------------------
    Device: {device}
    Using: {model_used}
    Model: {model.__class__.__name__}
    Pretrained: True
    Batch size: {batch_size}
    Number of epochs: {train_epochs}
    Learning Rate: {optimizer.param_groups[0]['lr']}
    Weight Decay: {optimizer.defaults.get('weight_decay', 'N/A')}
    Tiny ViT Completely Frozen: {freeze_tinyvit}

    Train JSON: {args.train_json_data}
    Eval JSON: {args.eval_json_data}
    Image Directory: {args.img_dir}

    Transformer Dropout Rate: {dropout_rate}

    Using GLAN: {True}
    GLAN Dropout Rate: {glan_dropout}
    GLAN Number of Layers: {glan_num_layers}
    GLAN Epochs: {glan_epochs}
    GLAN Learning Rate: {glan_lr}
    GLAN Early Stoppage: After {glan_early_stop} epochs of no accuracy improvement
    GLAN Weight Decay: {glan_weight_decay}
    GLAN Hidden Dimention: {glan_hidden_dim}

    Weighing Ambiguous Training Samples: {weigh_ambig_samples}
    Weight of Ambiguous Training Samples: {sampler_ambig_factor}

    Smoothing: {smoothing}
    Use Heatmap Mask: {use_heatmap_mask}
    Mask Sigma: {sigma_val}
    Scheduler: ReduceLROnPlateau, Mode: {scheduler_mode}, Factor: {scheduler_factor}, Patience: {scheduler_patience}
    ---------------------------------------------------------------------------
    """
    log_summary(summary)

    class EnsembleTrainer(Trainer):
        def run_model(self, num_epoch, train_loader, eval_loader, view, scheduler=None):
            best_epochs = 0  # for early stopping
            for epoch in range(num_epoch):
                epoch_start = torch.cuda.Event(enable_timing=True)
                epoch_end = torch.cuda.Event(enable_timing=True)
                epoch_start.record()
                running_loss = 0
                for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", unit="batch")):
                    loss = self.train(batch)
                    running_loss += loss * len(batch)

                epoch_train_loss = running_loss / len(train_loader)

                # eval raw and GNN accuracies
                raw_acc, gnn_acc, eval_loss = self.evaluate(eval_loader)
                log_summary(
                    f"Epoch {epoch} — Train Loss: {epoch_train_loss:.4f} | Raw Acc: {raw_acc:.4f} | GNN Acc: {gnn_acc:.4f} | Eval Loss: {eval_loss:.4f}"
                )

                # store metrics
                self.epoch_losses.append(epoch_train_loss)
                self.epoch_eval_losses.append(eval_loss)
                self.epoch_accuracies.append(gnn_acc)

                if scheduler is not None:
                    scheduler.step(gnn_acc)

                early_stoppage = glan_early_stop if args.use_glan else 10
                if gnn_acc > self.best_accuracy:
                    self.best_accuracy = save_best_model(
                        self.model, gnn_acc, self.best_accuracy,
                        filename=view + '_best_model.pth'
                    )
                    log_summary(f"New best model with {self.best_accuracy:.4f} accuracy saved to {view+'_best_model.pth'}")
                    best_epochs = 0
                else:
                    best_epochs += 1
                    if best_epochs >= early_stoppage:
                        log_summary(f"No improvement over {early_stoppage} epochs, stopping early with best model at {self.best_accuracy:.4f} accuracy.")
                        break

                log_summary(f"No Improvement count: {best_epochs}")

                epoch_end.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start.elapsed_time(epoch_end) / 1000.0  # seconds
                hours = int(epoch_time // 3600)
                minutes = int((epoch_time % 3600) // 60)
                seconds = int(epoch_time % 60)
                log_summary(f"Epoch {epoch} took {hours}h {minutes}m {seconds}s")

                if (epoch+1) % 2 == 0:
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        epoch_train_loss, gnn_acc,
                        filename=view + '_ckpt.pth'
                    )
                    log_summary(f"Checkpoint saved to {view+'_colorid_ckpt.pth'}")
            log_summary("\nTraining complete.")

    trainer = EnsembleTrainer(model=model, loss=criterion, optimizer=optimizer, device=device)

    trainer.run_model(
        num_epoch=train_epochs,
        train_loader=train_loader,
        eval_loader=eval_loader,
        view=f"ensemble_seed_{args.seed}",
        scheduler=scheduler
    )

    # save final ckpt
    checkpoint_path = os.path.join(args.output_dir, f"ckpt_seed_{args.seed}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    log_summary(f"Training complete for seed {args.seed}. Checkpoints and logs saved in {args.output_dir}")
    summary_fh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--train_json_data', type=str, required=True)
    parser.add_argument('--eval_json_data', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_glan', action='store_true', default=True)
    args = parser.parse_args()
    main(args) 