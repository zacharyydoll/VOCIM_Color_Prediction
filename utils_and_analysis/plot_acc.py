import re
import matplotlib.pyplot as plt

log_file = "/mydata/vocim/zachary/color_prediction/TinyViT_with_mask_GLAN/output_summary.log"
output_file = "accuracy.png"

train_accuracy = {}
val_accuracy = {}

with open(log_file, "r") as f:
    for line in f:
        m_val = re.search(r"Epoch (\d+),\s*Train Loss: [\d\.eE+-]+,\s*Eval Accuracy: ([\d\.eE+-]+),\s*Eval Loss: [\d\.eE+-]+", line, re.IGNORECASE)
        if m_val:
            epoch = int(m_val.group(1))
            val_acc = float(m_val.group(2))
            val_accuracy[epoch] = val_acc

        m_train = re.search(r"Epoch (\d+),\s*Train Accuracy: ([\d\.eE+-]+)", line, re.IGNORECASE)
        if m_train:
            epoch = int(m_train.group(1))
            train_acc = float(m_train.group(2))
            train_accuracy[epoch] = train_acc

if not train_accuracy:
    print("No training accuracy found in the log file. Only validation accuracy will be plotted.")

epochs_val = sorted(val_accuracy.keys())
val_acc_list = [val_accuracy[ep] for ep in epochs_val]

if train_accuracy:
    epochs_train = sorted(train_accuracy.keys())
    train_acc_list = [train_accuracy[ep] for ep in epochs_train]
else:
    epochs_train = []
    train_acc_list = []

plt.figure(figsize=(10, 6))
if train_acc_list:
    plt.plot(epochs_train, train_acc_list, marker='o', label='Training Accuracy')
plt.plot(epochs_val, val_acc_list, marker='o', label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(output_file)
plt.close()

print(f"Accuracy graph saved as {output_file}")
