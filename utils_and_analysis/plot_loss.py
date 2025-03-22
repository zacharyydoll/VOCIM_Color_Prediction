import re
import matplotlib.pyplot as plt

training_log_path = "../logs/output_summary.log"
output_file = "loss.png"

with open(training_log_path, "r") as f:
    log_data = f.read()

train_losses = {}
eval_losses = {}

for line in log_data.splitlines():
    m = re.search(r"Epoch (\d+),\s*Train Loss:\s*([\d\.eE+-]+),\s*Eval Accuracy:\s*[\d\.eE+-]+,\s*Eval Loss:\s*([\d\.eE+-]+)", line, re.IGNORECASE)
    if m:
        epoch = int(m.group(1))
        train_loss = float(m.group(2))
        eval_loss = float(m.group(3))
        if epoch not in train_losses:
            train_losses[epoch] = train_loss
        if epoch not in eval_losses:
            eval_losses[epoch] = eval_loss

epochs = sorted(train_losses.keys())
train_loss_list = [train_losses[ep] for ep in epochs]
eval_loss_list = [eval_losses[ep] for ep in epochs if ep in eval_losses]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_list, marker='o', label='Training Loss')
plt.plot(epochs, eval_loss_list, marker='o', label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(output_file)
plt.close()

print(f'Graph saved as {output_file}')
