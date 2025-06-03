import os
import re
import matplotlib.pyplot as plt
import numpy as np

ENSEMBLE_ROOT = "/mydata/vocim/zachary/color_prediction/deep_ensembles/ensembles"
OUTDIR = "/mydata/vocim/zachary/color_prediction/deep_ensembles"

def parse_log(log_path):
    epochs = []
    train_losses = []
    gnn_accs = []
    raw_accs = []
    eval_losses = []
    with open(log_path, "r") as f:
        for line in f:
            m = re.match(
                r"Epoch (\d+) — Train Loss: ([\d\.]+) \| Raw Acc: ([\d\.]+) \| GNN Acc: ([\d\.]+) \| Eval Loss: ([\d\.]+)",
                line)
            if m:
                epoch = int(m.group(1))
                train_loss = float(m.group(2))
                raw_acc = float(m.group(3))
                gnn_acc = float(m.group(4))
                eval_loss = float(m.group(5))
                epochs.append(epoch)
                train_losses.append(train_loss)
                raw_accs.append(raw_acc)
                gnn_accs.append(gnn_acc)
                eval_losses.append(eval_loss)
    return epochs, train_losses, raw_accs, gnn_accs, eval_losses

def main():
    model_dirs = [os.path.join(ENSEMBLE_ROOT, d) for d in os.listdir(ENSEMBLE_ROOT)
                  if os.path.isdir(os.path.join(ENSEMBLE_ROOT, d))]
    all_epochs = []
    all_train_losses = []
    all_gnn_accs = []
    all_raw_accs = []
    all_eval_losses = []
    labels = []

    for model_dir in sorted(model_dirs):
        log_path = os.path.join(model_dir, "logs", "output_summary.log")
        if not os.path.isfile(log_path):
            continue
        epochs, train_losses, raw_accs, gnn_accs, eval_losses = parse_log(log_path)
        all_epochs.append(epochs)
        all_train_losses.append(train_losses)
        all_gnn_accs.append(gnn_accs)
        all_raw_accs.append(raw_accs)
        all_eval_losses.append(eval_losses)
        labels.append(os.path.basename(model_dir))

    # --- Improved Accuracy Plot ---
    max_len = max(len(accs) for accs in all_gnn_accs)
    acc_matrix = np.full((len(all_gnn_accs), max_len), np.nan)
    for i, accs in enumerate(all_gnn_accs):
        acc_matrix[i, :len(accs)] = accs

    mean_acc = np.nanmean(acc_matrix, axis=0)
    std_acc = np.nanstd(acc_matrix, axis=0)
    epochs = np.arange(max_len)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_acc, color='black', label='Ensemble Mean', linewidth=2)
    plt.fill_between(epochs, mean_acc-std_acc, mean_acc+std_acc, color='gray', alpha=0.3, label='±1 Std')

    # Plot smoothed individual curves
    window = 5
    for accs, label in zip(all_gnn_accs, labels):
        if len(accs) >= window:
            smoothed = np.convolve(accs, np.ones(window)/window, mode='valid')
            plt.plot(epochs[:len(smoothed)], smoothed, alpha=0.5, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("GNN Accuracy")
    plt.title("GNN Accuracy Evolution (All Ensemble Members)")
    plt.ylim(0.96, 0.982)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ensemble_gnn_accuracy_evolution.png"))
    plt.close()

    # Plot train loss
    plt.figure(figsize=(10, 6))
    for epochs, train_losses, label in zip(all_epochs, all_train_losses, labels):
        plt.plot(epochs, train_losses, marker='o', label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss Evolution (All Ensemble Members)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ensemble_train_loss_evolution.png"))
    plt.close()

if __name__ == "__main__":
    main()