import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Path to your evaluation results pickle file
PKL_PATH = '/mydata/vocim/zachary/color_prediction/gnn_enhancement/weighted_sampler_prob_bird_nodes/ambig_res/eval_results/evaluation_metrics.pkl'

# Output directory for plots
outdir = 'confidence_report'
os.makedirs(outdir, exist_ok=True)

# 1. Load evaluation results
with open(PKL_PATH, 'rb') as f:
    metrics = pickle.load(f)

probs_list = metrics['probabilities']  # list of [num_classes] arrays
image_paths = metrics['image_paths']   # list of image paths

# 2. Compute per-bird confidence and entropy
per_bird_confidences = []
per_bird_entropies = []
per_frame_confidences = []
per_frame_entropies = []
frame_conf_map = {}
frame_entropy_map = {}

# Group by frame
from collections import defaultdict
frame_to_indices = defaultdict(list)
for i, pth in enumerate(image_paths):
    frame_id = pth.split('_bird')[0]
    frame_to_indices[frame_id].append(i)

for frame_id, idxs in frame_to_indices.items():
    frame_conf = []
    frame_entropy = []
    for i in idxs:
        probs = np.array(probs_list[i])
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

# 3. Plot histograms
plt.figure(figsize=(8,5))
plt.hist(per_bird_confidences, bins=30, alpha=0.7)
plt.xlabel('Per-bird Confidence (max prob)')
plt.ylabel('Count')
plt.title('Histogram of Per-bird Confidence')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'per_bird_confidence_hist.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.hist(per_frame_confidences, bins=30, alpha=0.7)
plt.xlabel('Per-frame Average Confidence')
plt.ylabel('Count')
plt.title('Histogram of Per-frame Average Confidence')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'per_frame_confidence_hist.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.hist(per_bird_entropies, bins=30, alpha=0.7)
plt.xlabel('Per-bird Entropy')
plt.ylabel('Count')
plt.title('Histogram of Per-bird Entropy')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'per_bird_entropy_hist.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.hist(per_frame_entropies, bins=30, alpha=0.7)
plt.xlabel('Per-frame Average Entropy')
plt.ylabel('Count')
plt.title('Histogram of Per-frame Average Entropy')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'per_frame_entropy_hist.png'))
plt.close()

# 3b. CDF plots
def plot_cdf(data, xlabel, title, filename):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.figure(figsize=(8,5))
    plt.plot(sorted_data, cdf)
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Fraction')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

# Per-bird confidence CDF
plot_cdf(per_bird_confidences, 'Per-bird Confidence (max prob)', 'CDF of Per-bird Confidence', 'per_bird_confidence_cdf.png')
# Per-frame confidence CDF
plot_cdf(per_frame_confidences, 'Per-frame Average Confidence', 'CDF of Per-frame Average Confidence', 'per_frame_confidence_cdf.png')
# Per-bird entropy CDF
plot_cdf(per_bird_entropies, 'Per-bird Entropy', 'CDF of Per-bird Entropy', 'per_bird_entropy_cdf.png')
# Per-frame entropy CDF
plot_cdf(per_frame_entropies, 'Per-frame Average Entropy', 'CDF of Per-frame Average Entropy', 'per_frame_entropy_cdf.png')

# 4. Save confidences for further analysis
with open('confidence_metrics.pkl', 'wb') as f:
    pickle.dump({
        'per_bird_confidences': per_bird_confidences,
        'per_frame_confidences': per_frame_confidences,
        'per_bird_entropies': per_bird_entropies,
        'per_frame_entropies': per_frame_entropies,
        'frame_conf_map': frame_conf_map,
        'frame_entropy_map': frame_entropy_map
    }, f)

print(f'Confidence analysis complete. Histograms saved in {outdir}/') 