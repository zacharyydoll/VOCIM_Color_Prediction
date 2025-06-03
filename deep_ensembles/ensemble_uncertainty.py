import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

ENSEMBLE_ROOT = "/mydata/vocim/zachary/color_prediction/deep_ensembles/ensembles"
OUTDIR = "/mydata/vocim/zachary/color_prediction/deep_ensembles"

def load_probs(key):
    all_probs = []
    for model_dir in sorted(os.listdir(ENSEMBLE_ROOT)):
        pkl_path = os.path.join(ENSEMBLE_ROOT, model_dir, "eval_results.pkl")
        if not os.path.isfile(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            metrics = pickle.load(f)
        arr = np.array(metrics[key])  # shape [num_samples, num_classes]
        all_probs.append(arr)
    return np.stack(all_probs, axis=0)  # shape [num_models, num_samples, num_classes]

def compute_epistemic(probs):
    # probs: [num_models, num_samples, num_classes]
    p_ens = np.mean(probs, axis=0)  # [num_samples, num_classes]
    H_ens = -np.sum(p_ens * np.log(p_ens + 1e-12), axis=1)  # [num_samples]
    H_exp = -np.sum(probs * np.log(probs + 1e-12), axis=2)  # [num_models, num_samples]
    H_exp_mean = np.mean(H_exp, axis=0)  # [num_samples]
    epistemic = H_ens - H_exp_mean
    return H_ens, H_exp_mean, epistemic

def plot_cdf(data, label, color):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label, color=color)

def main():
    for key, label in [("tinyvit_probabilities", "TinyViT"), ("gnn_soft_outputs", "GNN")]:
        probs = load_probs(key)
        H_ens, H_exp_mean, epistemic = compute_epistemic(probs)
        plt.figure()
        plot_cdf(H_ens, f"{label} Predictive Entropy", "blue")
        plot_cdf(H_exp_mean, f"{label} Expected Entropy", "green")
        plot_cdf(epistemic, f"{label} Epistemic (Mutual Info)", "red")
        plt.xlabel("Uncertainty")
        plt.ylabel("CDF")
        plt.legend()
        plt.title(f"{label} Uncertainty CDFs (Ensemble)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"ensemble_{label.lower()}_uncertainty_cdfs.png"))
        plt.close()

if __name__ == "__main__":
    main()