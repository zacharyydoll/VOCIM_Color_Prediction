import os
import re
import matplotlib.pyplot as plt
import numpy as np

ENSEMBLE_DIR = './ensembles'
MODEL_PREFIX = 'model'
SETS = ['normal', 'ambig']

def extract_accuracy(log_path):
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip().startswith('accuracy:'):
                return float(line.strip().split('accuracy:')[1].strip())
    return None

model_dirs = sorted([d for d in os.listdir(ENSEMBLE_DIR) if d.startswith(MODEL_PREFIX)])
accuracies = {s: [] for s in SETS}

for model in model_dirs:
    for s in SETS:
        log_path = os.path.join(ENSEMBLE_DIR, model, s, 'eval_results.log')
        if os.path.exists(log_path):
            acc = extract_accuracy(log_path)
            if acc is not None:
                accuracies[s].append(acc)
            else:
                accuracies[s].append(np.nan)
        else:
            accuracies[s].append(np.nan)

x = np.arange(len(model_dirs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 8))
rects1 = ax.bar(x - width/2, accuracies['normal'], width, label='Full')
rects2 = ax.bar(x + width/2, accuracies['ambig'], width, label='Ambig')

mean_normal = np.nanmean(accuracies['normal'])
mean_ambig = np.nanmean(accuracies['ambig'])
ax.axhline(mean_normal, color='blue', linestyle='--', label='Full Mean')
ax.axhline(mean_ambig, color='orange', linestyle='--', label='Ambig Mean')

ax.text(-0.5, mean_normal, f"{mean_normal:.4f}", color='blue', fontsize=10, va='center', ha='right', fontweight='bold', backgroundcolor='white')
ax.text(-0.5, mean_ambig, f"{mean_ambig:.4f}", color='orange', fontsize=10, va='center', ha='right', fontweight='bold', backgroundcolor='white')

ax.set_ylabel('Accuracy')
ax.set_xlabel('Ensemble Model')
ax.set_title('Ensemble Accuracies on Full and Ambiguous Test Sets')
ax.set_xticks(x)
ax.set_xticklabels(model_dirs)
ax.set_ylim(0,1)
ax.set_ylim(0.8, 1.0)
ax.legend()

for i, v in enumerate(accuracies['normal']):
    ax.text(i - width/2, v + 0.005, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
for i, v in enumerate(accuracies['ambig']):
    ax.text(i + width/2, v + 0.002, f"{v:.4f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/mydata/vocim/zachary/color_prediction/deep_ensembles/accuracies_histograms.png', dpi=200)
