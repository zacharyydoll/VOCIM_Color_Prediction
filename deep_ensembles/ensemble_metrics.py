import os
import re
import matplotlib.pyplot as plt
import numpy as np

ENSEMBLE_ROOT = "/mydata/vocim/zachary/color_prediction/deep_ensembles/ensembles"
OUTDIR       = "/mydata/vocim/zachary/color_prediction/deep_ensembles"

def parse_log(log_path):
    epochs = []
    train_losses = []
    raw_accs = []
    gnn_accs = []
    eval_losses = []
    with open(log_path, "r") as f:
        for line in f:
            m = re.match(
                r"Epoch (\d+) — Train Loss: ([\d\.]+) \| Raw Acc: ([\d\.]+) \| GNN Acc: ([\d\.]+) \| Eval Loss: ([\d\.]+)",
                line)
            if m:
                epoch      = int(m.group(1))
                train_loss = float(m.group(2))
                raw_acc    = float(m.group(3))
                gnn_acc    = float(m.group(4))
                eval_loss  = float(m.group(5))
                epochs.append(epoch)
                train_losses.append(train_loss)
                raw_accs.append(raw_acc)
                gnn_accs.append(gnn_acc)
                eval_losses.append(eval_loss)
    return epochs, train_losses, raw_accs, gnn_accs, eval_losses

def main():
    model_dirs = [
        os.path.join(ENSEMBLE_ROOT, d)
        for d in os.listdir(ENSEMBLE_ROOT)
        if os.path.isdir(os.path.join(ENSEMBLE_ROOT, d))
    ]

    all_epochs = []
    all_train_losses = []
    all_gnn_accs = []
    all_raw_accs = []
    all_eval_losses = []
    labels = []

    # parse each seed’s log
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


    max_len_acc = max(len(accs) for accs in all_gnn_accs)
    acc_matrix = np.full((len(all_gnn_accs), max_len_acc), np.nan)
    for i, accs in enumerate(all_gnn_accs):
        acc_matrix[i, :len(accs)] = accs

    mean_acc = np.nanmean(acc_matrix, axis=0)
    std_acc  = np.nanstd(acc_matrix, axis=0)
    epochs_acc = np.arange(max_len_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_acc, mean_acc, color='black', label='Ensemble Mean', linewidth=2)
    plt.fill_between(epochs_acc, mean_acc - std_acc, mean_acc + std_acc, color='gray', alpha=0.3, label='±1 Std')

    window = 5
    for accs, label in zip(all_gnn_accs, labels):
        if len(accs) >= window:
            smoothed = np.convolve(accs, np.ones(window)/window, mode='valid')
            plt.plot(epochs_acc[:len(smoothed)], smoothed, alpha=0.5, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("GNN Accuracy")
    plt.title("GNN Accuracy Evolution (All Ensemble Members)")
    plt.ylim(0.96, 0.982)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ensemble_gnn_accuracy_evolution.png"))
    plt.close()

    max_len_loss = max(len(losses) for losses in all_train_losses)

    train_matrix = np.full((len(all_train_losses), max_len_loss), np.nan)
    for i, losses in enumerate(all_train_losses):
        train_matrix[i, :len(losses)] = losses

    eval_matrix = np.full((len(all_eval_losses), max_len_loss), np.nan)
    for i, losses in enumerate(all_eval_losses):
        eval_matrix[i, :len(losses)] = losses

    mean_eval_loss = np.nanmean(eval_matrix, axis=0)
    epochs_loss = np.arange(max_len_loss)

    plt.figure(figsize=(10, 6))

    for epochs, train_losses, label in zip(all_epochs, all_train_losses, labels):
        plt.plot(epochs, train_losses, marker='o', label=f"{label} (train)")

    plt.plot(epochs_loss, mean_eval_loss,
             color='red',
             linestyle='--',
             linewidth=2,
             label="Ensemble Mean (val loss)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss (per-seed) and Mean Validation Loss (Ensemble)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ensemble_train_plus_mean_val_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for epochs, eval_losses, label in zip(all_epochs, all_eval_losses, labels):
        plt.plot(epochs, eval_losses, marker='x', linestyle='--', label=f"{label} (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Evolution (All Ensemble Members)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ensemble_eval_loss_evolution.png"))
    plt.close()

if __name__ == "__main__":
    main()
