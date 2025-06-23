# deep_ensembles/train_ensemble.py

import sys
import os
import argparse
import torch
import numpy as np
import random

# ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Trainer
from model_builder import build_model
from dataloader import get_train_dataloader, get_eval_dataloader
from utils import save_checkpoint, save_best_model

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Import all hyperparameters (including our new ensemble_unfreeze_all)
from config import (
    batch_size, num_epochs, dropout_rate, learning_rate, weight_decay,
    scheduler_mode, scheduler_factor, scheduler_patience, num_classes, model_name,
    glan_early_stop, glan_weight_decay, smoothing, sigma_val, use_heatmap_mask,
    glan_hidden_dim, model_used, glan_dropout, glan_lr, glan_epochs, freeze_tinyvit,
    glan_num_layers, weigh_ambig_samples, sampler_ambig_factor,
    ensemble_unfreeze_all,  
)

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

    model = build_model(pretrained=True, dropout_rate=dropout_rate, num_classes=num_classes)
    model = model.to(device)

    pretrained_ckpt = "/mydata/vocim/zachary/color_prediction/model_archives/TinyViT_with_mask/top_colorid_best_model_9831v_9765t.pth"
    if os.path.exists(pretrained_ckpt):
        print(f"Loading pretrained TinyViT from {pretrained_ckpt}")
        tmp = Trainer(model=model, loss=nn.NLLLoss(), optimizer=None, device=device)
        tmp.load_model(pretrained_ckpt)
        del tmp
    else:
        print(f"Warning: checkpoint not found at {pretrained_ckpt}, training from scratch")

    if ensemble_unfreeze_all:
        # unfreeze everything
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = list(model.parameters())
    else:
        # freeze all first
        for p in model.parameters():
            p.requires_grad = False

        # unfreeze GNN if we have it
        if args.use_glan and hasattr(model, 'color_gnn'):
            gnn_params = list(model.color_gnn.parameters())
            for p in gnn_params:
                p.requires_grad = True
        else:
            gnn_params = []

        # unfreeze last transformer block + head
        if model_used.lower() == "tinyvit":
            last_stage = model.stages[-1]
            last_block = last_stage.blocks[-1]
            backbone_params = list(last_block.parameters())
        else:
            backbone_params = list(model.parameters())

        if hasattr(model, 'head'):
            head_params = list(model.head.parameters())
        elif hasattr(model, 'classifier'):
            head_params = list(model.classifier.parameters())
        else:
            head_params = []

        for p in backbone_params + head_params:
            p.requires_grad = True

        trainable_params = gnn_params + backbone_params + head_params

    # optimizer over only unfrozen parameters
    optimizer = AdamW(
        trainable_params,
        lr=(learning_rate if ensemble_unfreeze_all else glan_lr),
        weight_decay=glan_weight_decay if ensemble_unfreeze_all else glan_weight_decay
    )

    # LR scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True
    )

    ambiguous_json_path = "data/ambig_train_samples.json"
    train_loader = get_train_dataloader(
        args.train_json_data,
        args.img_dir,
        batch_size=batch_size,
        ambiguous_json_path=ambiguous_json_path,
        ambiguous_factor=sampler_ambig_factor
    )
    eval_loader = get_eval_dataloader(args.eval_json_data, args.img_dir, batch_size=batch_size)

    checkpoint_dir = os.path.abspath(args.output_dir)
    logs_dir = os.path.abspath(os.path.join(checkpoint_dir, '..', 'logs'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    output_log_file = os.path.join(logs_dir, "output.log")
    summary_log_file = os.path.join(logs_dir, "output_summary.log")

    # redirect stdout/stderr
    sys.stdout = open(output_log_file, 'w')
    sys.stderr = sys.stdout
    summary_fh = open(summary_log_file, 'w')

    def log_summary(message):
        print(message)
        summary_fh.write(message + "\n")
        summary_fh.flush()

    model_type = model.__class__.__name__
    summary = f"""
    Training Summary (Ensemble member seed={args.seed}):
    ---------------------------------------------------------------------------
    Device: {device}
    Using model type: {model_used} → {model_type}
    Pretrained: True
    Batch size: {batch_size}
    Number of epochs: {train_epochs}
    Learning Rate: {optimizer.param_groups[0]['lr']}
    Weight Decay: {optimizer.defaults.get('weight_decay', 'N/A')}
    
    ensemble_unfreeze_all: {ensemble_unfreeze_all}

    Train JSON: {args.train_json_data}
    Eval JSON: {args.eval_json_data}
    Image Directory: {args.img_dir}

    Transformer Dropout Rate: {dropout_rate}
    Using GLAN: {args.use_glan}
    GLAN Dropout Rate: {glan_dropout}
    GLAN Number of Layers: {glan_num_layers}
    GLAN Epochs: {glan_epochs}
    GLAN Learning Rate: {glan_lr}
    GLAN Early Stoppage: After {glan_early_stop} epochs no improvement
    GLAN Weight Decay: {glan_weight_decay}
    GLAN Hidden Dimension: {glan_hidden_dim}

    Weighing Ambiguous Samples: {weigh_ambig_samples}
    Ambiguous Sample Factor: {sampler_ambig_factor}

    Label Smoothing: {smoothing}
    Use Heatmap Mask: {use_heatmap_mask}
    Mask Sigma: {sigma_val}

    Scheduler: ReduceLROnPlateau, Mode={scheduler_mode}, Factor={scheduler_factor}, Patience={scheduler_patience}
    All checkpoints (best + ckpt) will be saved under: {checkpoint_dir}
    ---------------------------------------------------------------------------
    """
    log_summary(summary)

    class EnsembleTrainer(Trainer):
        def __init__(self, model, optimizer, loss, device, checkpoint_dir):
            super().__init__(model, optimizer, loss, device)
            self.checkpoint_dir = checkpoint_dir

        def run_model(self, num_epoch, train_loader, eval_loader, view, scheduler=None):
            best_epochs = 0
            for epoch in range(num_epoch):
                epoch_start = torch.cuda.Event(enable_timing=True)
                epoch_end = torch.cuda.Event(enable_timing=True)
                epoch_start.record()

                running_loss = 0
                for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", unit="batch")):
                    loss = self.train(batch)
                    running_loss += loss * len(batch)

                epoch_train_loss = running_loss / len(train_loader)

                # Evaluate raw + GNN accuracy
                raw_acc, gnn_acc, eval_loss = self.evaluate(eval_loader)
                log_summary(
                    f"Epoch {epoch} — Train Loss: {epoch_train_loss:.4f} | "
                    f"Raw Acc: {raw_acc:.4f} | GNN Acc: {gnn_acc:.4f} | Eval Loss: {eval_loss:.4f}"
                )

                self.epoch_losses.append(epoch_train_loss)
                self.epoch_eval_losses.append(eval_loss)
                self.epoch_accuracies.append(gnn_acc)

                if scheduler is not None:
                    scheduler.step(gnn_acc)

                early_stoppage = glan_early_stop if args.use_glan else 10
                best_filename = os.path.join(self.checkpoint_dir, view + "_best_model.pth")
                if gnn_acc > self.best_accuracy:
                    self.best_accuracy = save_best_model(
                        self.model, gnn_acc, self.best_accuracy,
                        filename=best_filename
                    )
                    log_summary(f"New best model with {self.best_accuracy:.4f} accuracy saved to {best_filename}")
                    best_epochs = 0
                else:
                    best_epochs += 1
                    if best_epochs >= early_stoppage:
                        log_summary(
                            f"No improvement over {early_stoppage} epochs, "
                            f"early stopping with best={self.best_accuracy:.4f}."
                        )
                        break
                log_summary(f"No Improvement count: {best_epochs}")

                epoch_end.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start.elapsed_time(epoch_end) / 1000.0  # convert ms→s
                hours = int(epoch_time // 3600)
                minutes = int((epoch_time % 3600) // 60)
                seconds = int(epoch_time % 60)
                log_summary(f"Epoch {epoch} took {hours}h {minutes}m {seconds}s")

                # Save periodic checkpoint (every 2 epochs) under checkpoint_dir
                if (epoch + 1) % 2 == 0:
                    ckpt_filename = os.path.join(self.checkpoint_dir, view + "_ckpt.pth")
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        epoch_train_loss, gnn_acc,
                        filename=ckpt_filename
                    )
                    log_summary(f"Intermediate checkpoint saved to {ckpt_filename}")

            log_summary("\nTraining complete.")

    # Instantiate and run the ensemble trainer
    trainer = EnsembleTrainer(
        model=model,
        loss=nn.NLLLoss(),
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    trainer.run_model(
        num_epoch=train_epochs,
        train_loader=train_loader,
        eval_loader=eval_loader,
        view=f"ensemble_seed_{args.seed}",
        scheduler=scheduler
    )

    final_name = os.path.join(checkpoint_dir, f"final_state_dict_seed_{args.seed}.pth")
    torch.save(model.state_dict(), final_name)
    log_summary(f"Training complete for seed {args.seed}. Final state_dict saved to {final_name}")
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
