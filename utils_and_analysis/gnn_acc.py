import re
import matplotlib.pyplot as plt

logfile = "/mydata/vocim/zachary/color_prediction/gnn_enhancement/summary_3_layers.log"

epochs = []
raw_acc = []
gnn_acc = []

with open(logfile, "r") as f:
    for line in f:
        # Match lines like: Epoch 0 — Train Loss: ... | Raw Acc: 0.9813  |  GNN Acc: 0.9710  |  Eval Loss: 0.1274
        m = re.match(r"Epoch (\d+).*Raw Acc: ([0-9.]+).*GNN Acc: ([0-9.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            raw_acc.append(float(m.group(2)))
            gnn_acc.append(float(m.group(3)))

plt.figure(figsize=(8, 6))
plt.plot(epochs, raw_acc, label='Raw Accuracy (TinyViT)')
plt.plot(epochs, gnn_acc, label='GNN Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Raw vs GNN Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('gnn_acc.png')
plt.close()
print("Saved gnn_acc.png")