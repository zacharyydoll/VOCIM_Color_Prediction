import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from tqdm import tqdm
import os
import sys
import yaml
import json
import re

def setup_logging(results_dir):
    """Set up logging to both console and file."""
    log_path = os.path.join(results_dir, 'linear_assignment_eval.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w')
    
    class Logger:
        def __init__(self, log_file):
            self.log_file = log_file
            
        def write(self, message):
            sys.__stdout__.write(message)
            self.log_file.write(message)
            
        def flush(self):
            sys.__stdout__.flush()
            self.log_file.flush()
    
    sys.stdout = Logger(log_file)
    return log_file

def load_predictions(results_dir):
    """Load predictions from evaluation metrics pickle file."""
    metrics_file = os.path.join(results_dir, 'eval_results.pkl')
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def evaluate_with_linear_assignment(metrics):
    """Evaluate predictions using linear assignment with softmax probabilities."""
    if 'tinyvit_probabilities' in metrics:
        probabilities = metrics['tinyvit_probabilities']
    elif 'probabilities' in metrics:
        probabilities = metrics['probabilities']
    else:
        raise KeyError("Could not find a probabilities key ('tinyvit_probabilities' or 'probabilities') in the metrics file.")
        
    labels = metrics['labels']
    image_paths = metrics['image_paths']
    
    # Group predictions by frame
    frame_groups = {}
    for i, img_path in enumerate(image_paths):
        frame_id = re.sub(r'_bird_\d+\.png$', '', img_path)
        if frame_id not in frame_groups:
            frame_groups[frame_id] = []
        frame_groups[frame_id].append(i)
    
    total_correct = 0
    total_assigned = 0
    assigned_labels = []
    true_labels_for_assigned = []
    
    # process each frame
    for frame_id, indices in tqdm(frame_groups.items(), desc="Processing Frames"):
        frame_probs = np.array([probabilities[i] for i in indices])
        frame_labels = [labels[i] for i in indices]
        
        num_birds = len(frame_probs)
        if num_birds == 0:
            continue
        
        # sum probs to find the most likely colors for this frame
        color_sums = frame_probs.sum(axis=0)
        
        # select top-k colors, where k is the number of birds
        k = min(num_birds, len(color_sums))
        top_k_color_indices = np.argsort(color_sums)[-k:][::-1]
        
        # reduced probability matrix
        reduced_probs = frame_probs[:, top_k_color_indices]
        
        cost_matrix = 1 - reduced_probs
        
        # apply Hungarian algo
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # map assigned column indices back to original color indices
        assigned_colors = [top_k_color_indices[j] for j in col_ind]
        
        for i, original_idx in enumerate(row_ind):
            assigned_labels.append(assigned_colors[i])
            true_labels_for_assigned.append(frame_labels[original_idx])
            if assigned_colors[i] == frame_labels[original_idx]:
                total_correct += 1
        
        total_assigned += len(row_ind)

    accuracy = total_correct / total_assigned if total_assigned > 0 else 0
    
    return accuracy, assigned_labels, true_labels_for_assigned


def process_results_directory(results_dir):
    """Process all results in the directory."""
    metrics = load_predictions(results_dir)
    
    direct_accuracy = accuracy_score(metrics['labels'], metrics['predictions'])
    
    la_accuracy, la_preds, la_labels = evaluate_with_linear_assignment(metrics)
    
    print("\n" + "="*50)
    print(f"       Results for Directory: {results_dir}")
    print("="*50 + "\n")
    
    print(f"Direct Evaluation Accuracy (per-crop): {direct_accuracy:.4f}")
    print(f"Linear Assignment Accuracy (per-frame, leak-free): {la_accuracy:.4f}")
    
    print("\n--- Direct Evaluation Report ---")
    print(classification_report(metrics['labels'], metrics['predictions'], zero_division=0))
    
    print("\n--- Linear Assignment Report ---")
    print(classification_report(la_labels, la_preds, zero_division=0))
    
    print("\nConfusion Matrix (Linear Assignment):")
    print(confusion_matrix(la_labels, la_preds))
    print("\n" + "="*50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "/mydata/vocim/zachary/color_prediction/lin_asg_ambig_res"
        print(f"Warning: No results directory provided. Using default: {results_dir}")

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        sys.exit(1)
        
    log_file = setup_logging(results_dir)
    
    try:
        process_results_directory(results_dir)
    finally:
        log_file.close()
        sys.stdout = sys.__stdout__ 