import re
import matplotlib.pyplot as plt

logfile = "/mydata/vocim/zachary/color_prediction/gnn_enhancement/summary_3_layers.log"

epochs = []
train_loss = []
eval_loss = []

with open(logfile, "r") as f:
    for line in f:
        # Match lines like: Epoch 0 — Train Loss: 1.1005  |  Raw Acc: 0.9813  |  GNN Acc: 0.9710  |  Eval Loss: 0.1274
        m = re.match(r"Epoch (\d+).*Train Loss: ([0-9.]+).*Eval Loss: ([0-9.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            eval_loss.append(float(m.group(3)))

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, eval_loss, label='Eval Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Eval Loss')
plt.legend()
plt.grid(True)
plt.savefig('gnn_loss.png')
plt.close()
print("Saved gnn_loss.png")