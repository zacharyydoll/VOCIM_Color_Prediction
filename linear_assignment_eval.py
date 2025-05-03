import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from tqdm import tqdm
import os
from config import num_classes
import sys
import yaml
import json
import re

def setup_logging(results_dir):
    """Set up logging to both console and file."""
    log_path = os.path.join(results_dir, 'linear_assignment_eval.log')
    log_file = open(log_path, 'w')
    
    class Logger:
        def __init__(self, log_file):
            self.log_file = log_file
            
        def write(self, message):
            self.log_file.write(message)
            sys.stdout.write(message)
            
        def flush(self):
            self.log_file.flush()
            sys.stdout.flush()
    
    sys.stdout = Logger(log_file)
    return log_file

def load_predictions(results_dir):
    """Load predictions from evaluation metrics pickle file."""
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.pkl')
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def get_available_colors(image_path, bird_identity_yaml, colormap_yaml):
    """Get all backpack colors present in an image."""
    # Extract the full path components
    # Example: /mydata/vocim/zachary/data/cropped/VOCIM_juvExpBP05/labeled-data_topview/BP_2023-06-30_06-07-02_808380_0020000/img04168_bird_1.png
    path_parts = image_path.split('/')
    
    # Find the experiment directory (e.g., VOCIM_juvExpBP05)
    exp_dir = None
    for part in path_parts:
        if part.startswith('VOCIM_'):
            exp_dir = part
            break
    
    if exp_dir is None:
        raise ValueError(f"Could not find experiment directory in path: {image_path}")
    
    # Get the view type (e.g., labeled-data_topview)
    view_type = path_parts[-3]
    
    # Get the BP directory name
    bp_dir = path_parts[-2]
    
    # Construct the full path as it appears in the YAML
    yaml_path = f"{exp_dir}/{view_type}/{bp_dir}"
    
    # Load bird identity mapping
    with open(bird_identity_yaml, 'r') as f:
        bird_identity_mapping = yaml.safe_load(f)
    
    # Load color map
    with open(colormap_yaml, 'r') as f:
        color_map = yaml.safe_load(f)
    
    # Get colors for all birds in this directory
    if yaml_path in bird_identity_mapping:
        bird_mapping = bird_identity_mapping[yaml_path]
    else:
        # Try to find a matching directory by removing the frame number
        # For example, "BP_2023-06-30_06-07-02_808380_0020000" -> "BP_2023-06-30_06-07-02_808380"
        base_bp_dir = '_'.join(bp_dir.split('_')[:-1])
        base_yaml_path = f"{exp_dir}/{view_type}/{base_bp_dir}"
        if base_yaml_path in bird_identity_mapping:
            bird_mapping = bird_identity_mapping[base_yaml_path]
        else:
            raise ValueError(f"Directory {yaml_path} not found in bird identity mapping.")
    
    # Get numeric labels for all colors
    colors = set()
    for color_name in bird_mapping.values():
        if color_name in color_map:
            colors.add(color_map[color_name])
    
    return list(colors)

def evaluate_with_linear_assignment(metrics, bird_identity_yaml, colormap_yaml):
    """Evaluate predictions using linear assignment with softmax probabilities."""
    predictions = metrics['predictions']
    probabilities = metrics['probabilities']  # Get softmax probabilities
    labels = metrics['labels']
    image_paths = metrics['image_paths']
    
    total_correct = 0
    total_images = len(predictions)
    
    # Group predictions by frame (remove bird number from path)
    frame_groups = {}
    for probs, pred, label, img_path in zip(probabilities, predictions, labels, image_paths):
        # Extract frame path by removing bird number
        frame_path = re.sub(r'_bird_\d+\.png$', '.png', img_path)
        if frame_path not in frame_groups:
            frame_groups[frame_path] = {
                'probs': [],
                'preds': [],
                'labels': [],
                'img_paths': []
            }
        frame_groups[frame_path]['probs'].append(probs)
        frame_groups[frame_path]['preds'].append(pred)
        frame_groups[frame_path]['labels'].append(label)
        frame_groups[frame_path]['img_paths'].append(img_path)
    
    # For each frame
    for frame_path, group in frame_groups.items():
        # Get all available colors in this frame (use first image path)
        available_colors = get_available_colors(group['img_paths'][0], bird_identity_yaml, colormap_yaml)
        n_predictions = len(group['probs'])
        n_colors = len(available_colors)
        
        if n_predictions > n_colors:
            continue
        
        # Create cost matrix using softmax probabilities
        # Rows: predictions, Columns: available colors
        cost_matrix = np.ones((n_predictions, n_colors))
        for i, probs in enumerate(group['probs']):
            for j, color in enumerate(available_colors):
                cost_matrix[i, j] = 1 - probs[color]  # Lower cost for higher probability
        
        # Use Hungarian algorithm to find best matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Get the assigned colors
        assigned_colors = [available_colors[j] for j in col_ind]
        
        # Count correct assignments
        for i, assigned_color in enumerate(assigned_colors):
            if assigned_color == group['labels'][i]:
                total_correct += 1
    
    accuracy = total_correct / total_images
    return accuracy

def process_results_directory(results_dir, bird_identity_yaml, colormap_yaml):
    """Process all results in the directory."""
    metrics = load_predictions(results_dir)
    
    # Calculate direct evaluation accuracy
    direct_accuracy = np.mean(np.array(metrics['predictions']) == np.array(metrics['labels']))
    
    # Calculate linear assignment accuracy
    la_accuracy = evaluate_with_linear_assignment(metrics, bird_identity_yaml, colormap_yaml)
    
    # Print results
    print(f"\nDirect Evaluation Accuracy: {direct_accuracy:.4f}")
    print(f"Linear Assignment Accuracy: {la_accuracy:.4f}")
    
    # Print confusion matrix and classification report for direct evaluation
    print("\nConfusion Matrix (Direct Evaluation):")
    print(confusion_matrix(metrics['labels'], metrics['predictions']))
    print("\nClassification Report (Direct Evaluation):")
    print(classification_report(metrics['labels'], metrics['predictions']))

if __name__ == "__main__":
    results_dir = "/mydata/vocim/zachary/color_prediction/TinyViT_with_mask_GLAN/report_normal_test_set"
    bird_identity_yaml = "newdata_bird_identity.yaml"
    colormap_yaml = "newdata_colormap.yaml"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        sys.exit(1)
    
    if not os.path.exists(bird_identity_yaml):
        print(f"Error: Bird identity YAML file '{bird_identity_yaml}' not found.")
        sys.exit(1)
    
    if not os.path.exists(colormap_yaml):
        print(f"Error: Colormap YAML file '{colormap_yaml}' not found.")
        sys.exit(1)
    
    process_results_directory(results_dir, bird_identity_yaml, colormap_yaml) 