#!/usr/bin/env python3
import json
from collections import Counter
import argparse
import re
import os
import yaml

BIRD_IDENTITY_YAML = "/mydata/vocim/zachary/color_prediction/newdata_bird_identity.yaml"
COLORMAP_YAML = "/mydata/vocim/zachary/color_prediction/newdata_colormap.yaml"

# load YAML mappings
with open(BIRD_IDENTITY_YAML, "r") as f:
    bird_identity_mapping = yaml.safe_load(f)
with open(COLORMAP_YAML, "r") as f:
    color_map = yaml.safe_load(f)

def extract_effective_label(annotation, img_paths):
    """
    Computes the effective numeric label for an annotation by:
      1. Parsing the identity string to get the bird key.
      2. Using the corresponding image's directory to look up the bird key in the bird identity YAML.
      3. Mapping the resulting color name to a numeric label using the colormap YAML.
    
    Returns:
        label (int): Effective numeric label.
    """
    identity_str = annotation['identity']  # e.g. "bird_y_1" or "bird_b_2"
    m = re.search(r'(bird(?:_[a-z])?_) *(\d+)', identity_str, re.IGNORECASE)
    if m:
        bird_key = f"bird_{m.group(2)}"
    else:
        raise ValueError(f"Could not parse identity from: {identity_str}")

    file_name = img_paths[annotation['image_id']]['file_name']
    directory = os.path.dirname(file_name)

    if directory not in bird_identity_mapping:
        raise ValueError(f"Directory {directory} not found in bird identity mapping.")
    bird_mapping = bird_identity_mapping[directory]

    if bird_key not in bird_mapping:
        raise ValueError(f"Bird key {bird_key} not found for directory {directory}.")

    color_name = bird_mapping[bird_key]

    if color_name not in color_map:
        raise ValueError(f"Color {color_name} not found in colormap.")
    label = color_map[color_name]
    return label

def count_effective_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    labels = [extract_effective_label(anno, data['images']) for anno in data['annotations']]
    counts = Counter(labels)
    return counts

def main(train_path, val_path, test_path):
    for split_name, path in zip(["Train", "Validation", "Test"], [train_path, val_path, test_path]):
        print(f"\nEffective label distribution for {split_name} file: {path}")
        counts = count_effective_labels(path)
        for label, cnt in sorted(counts.items()):
            print(f"Class {label}: {cnt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count effective label distribution for split JSON files")
    parser.add_argument("--train_json", type=str,
                        default="/mydata/vocim/zachary/color_prediction/data/newdata_cls_train_vidsplit_n.json",
                        help="Path to the training JSON file")
    parser.add_argument("--val_json", type=str,
                        default="/mydata/vocim/zachary/color_prediction/data/newdata_cls_val_vidsplit_n.json",
                        help="Path to the validation JSON file")
    parser.add_argument("--test_json", type=str,
                        default="/mydata/vocim/zachary/color_prediction/data/newdata_test_vidsplit_n.json",
                        help="Path to the test JSON file")
    args = parser.parse_args()
    
    main(args.train_json, args.val_json, args.test_json)
