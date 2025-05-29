import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Path to your evaluation results pickle file
PKL_PATH = '/mydata/vocim/zachary/color_prediction/eval_results/evaluation_metrics.pkl'

# Output directory for plots
outdir = 'confidence_report'
os.makedirs(outdir, exist_ok=True)

# 1. Load evaluation results
with open(PKL_PATH, 'rb') as f:
    metrics = pickle.load(f)

def get_confidence_and_entropy(prob_list, image_paths):
    per_bird_confidences = []
    per_bird_entropies = []
    per_frame_confidences = []
    per_frame_entropies = []
    frame_conf_map = {}
    frame_entropy_map = {}
    from collections import defaultdict
    frame_to_indices = defaultdict(list)
    for i, pth in enumerate(image_paths):
        frame_id = pth.split('_bird')[0]
        frame_to_indices[frame_id].append(i)
    for frame_id, idxs in frame_to_indices.items():
        frame_conf = []
        frame_entropy = []
        for i in idxs:
            probs = np.array(prob_list[i])
            conf = np.max(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            per_bird_confidences.append(conf)
            per_bird_entropies.append(entropy)
            frame_conf.append(conf)
            frame_entropy.append(entropy)
        avg_conf = np.mean(frame_conf)
        avg_entropy = np.mean(frame_entropy)
        per_frame_confidences.append(avg_conf)
        per_frame_entropies.append(avg_entropy)
        frame_conf_map[frame_id] = avg_conf
        frame_entropy_map[frame_id] = avg_entropy
    return (per_bird_confidences, per_frame_confidences, per_bird_entropies, per_frame_entropies)

def plot_cdf_with_quantile(data_dict, xlabel, title, filename, quantile=0.10, is_entropy=False):
    plt.figure(figsize=(8,5))
    quantile_texts = []
    # For entropy, use 1-quantile (top quantile)
    q = 1-quantile if is_entropy else quantile
    for label, data in data_dict.items():
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=label)
        x_q = np.percentile(sorted_data, q*100)
        y_q = q
        idx_q = np.searchsorted(cdf, y_q)
        x_q_actual = sorted_data[idx_q] if idx_q < len(sorted_data) else sorted_data[-1]
        plt.axvline(x=x_q, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=y_q, color='red', linestyle='--', alpha=0.7)
        plt.plot([x_q], [y_q], 'ro')
        quantile_texts.append(f"{label}: ({x_q:.3f}, {y_q:.2f})")
    # Place the quantile info in a box in the lower left
    quantile_info = "\n".join(quantile_texts)
    plt.legend(loc='upper right')
    plt.gca().text(
        0.1, 0.15, quantile_info,  # lower left
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', lw=0.5)
    )
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Fraction')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

image_paths = metrics['image_paths']

# TinyViT confidence/entropy
if 'tinyvit_probabilities' in metrics:
    tinyvit_probs = np.array(metrics['tinyvit_probabilities'])
    (tinyvit_bird_conf, tinyvit_frame_conf, tinyvit_bird_ent, tinyvit_frame_ent) = get_confidence_and_entropy(tinyvit_probs, image_paths)
else:
    raise RuntimeError('tinyvit_probabilities not found in metrics!')

# GNN confidence/entropy
if 'gnn_soft_outputs' in metrics:
    gnn_logits = np.array(metrics['gnn_soft_outputs'])
    exp_logits = np.exp(gnn_logits - np.max(gnn_logits, axis=1, keepdims=True))
    gnn_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    (gnn_bird_conf, gnn_frame_conf, gnn_bird_ent, gnn_frame_ent) = get_confidence_and_entropy(gnn_probs, image_paths)
else:
    raise RuntimeError('gnn_soft_outputs not found in metrics!')

# Plot CDFs for per-bird confidence
plot_cdf_with_quantile(
    {'TinyViT': tinyvit_bird_conf, 'GNN': gnn_bird_conf},
    'Per-bird Confidence (max prob)',
    'CDF of Per-bird Confidence',
    'cdf_per_bird_confidence.png',
    quantile=0.10,
    is_entropy=False
)

# Plot CDFs for per-frame average confidence
plot_cdf_with_quantile(
    {'TinyViT': tinyvit_frame_conf, 'GNN': gnn_frame_conf},
    'Per-frame Average Confidence',
    'CDF of Per-frame Average Confidence',
    'cdf_per_frame_confidence.png',
    quantile=0.10,
    is_entropy=False
)

# Plot CDFs for per-bird entropy (use 1-quantile)
plot_cdf_with_quantile(
    {'TinyViT': tinyvit_bird_ent, 'GNN': gnn_bird_ent},
    'Per-bird Entropy',
    'CDF of Per-bird Entropy',
    'cdf_per_bird_entropy.png',
    quantile=0.10,
    is_entropy=True
)

# Plot CDFs for per-frame average entropy (use 1-quantile)
plot_cdf_with_quantile(
    {'TinyViT': tinyvit_frame_ent, 'GNN': gnn_frame_ent},
    'Per-frame Average Entropy',
    'CDF of Per-frame Average Entropy',
    'cdf_per_frame_entropy.png',
    quantile=0.10,
    is_entropy=True
)

print(f'Confidence and entropy CDFs with quantile annotations saved in {outdir}/') 