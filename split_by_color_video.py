import json
import os
import copy
import random
import math
import yaml
import re 

bird_identity_yaml = "/mydata/vocim/zachary/color_prediction/newdata_bird_identity.yaml"
colormap_yaml = "/mydata/vocim/zachary/color_prediction/newdata_colormap.yaml"

with open(bird_identity_yaml, "r") as f:
    bird_identity_mapping = yaml.safe_load(f)
with open(colormap_yaml, "r") as f:
    color_map = yaml.safe_load(f)

def get_effective_color(anno, img_paths):
    """
    Extracts the effective color name for an annotation by:
      1. Parsing the identity string to get the bird key.
      2. Looking up the bird key in the YAML mapping based on the image's directory.
      
    Returns:
        color_name (str): The color associated with this bird.
    """
    identity_str = anno['identity']
    m = re.search(r'(bird(?:_[a-z])?_) *(\d+)', identity_str, re.IGNORECASE)
    if m:
        bird_key = f"bird_{m.group(2)}"
    else:
        raise ValueError(f"Could not parse identity from: {identity_str}")
    
    file_name = img_paths[anno['image_id']]['file_name']
    directory = os.path.dirname(file_name)
    
    if directory not in bird_identity_mapping:
        raise ValueError(f"Directory {directory} not found in bird identity mapping.")
    bird_mapping = bird_identity_mapping[directory]
    
    if bird_key not in bird_mapping:
        raise ValueError(f"Bird key {bird_key} not found for directory {directory}.")
    
    color_name = bird_mapping[bird_key]
    return color_name

def split_list(items, ratio):
    """
    Shuffle and split a list of items into three parts according to ratio.
    Expects ratio to be a list of three floats that sum to ~1.
    """
    assert len(ratio) == 3, "Ratio must have three elements (e.g., [0.7, 0.2, 0.1])."
    random.shuffle(items)
    n = len(items)
    train_n = int(n * ratio[0])
    val_n = int(n * ratio[1])
    train = items[:train_n]
    val = items[train_n:train_n + val_n]
    test = items[train_n + val_n:]
    if n >= 3:
        if len(train) == 0: train = [items[0]]
        if len(val) == 0: val = [items[1]]
        if len(test) == 0: test = [items[-1]]
    return train, val, test

def save_split(data, indices, output_name):
    """
    Creates a new JSON from the original master JSON data using the given annotation indices,
    and reassigns image and annotation IDs.
    """
    split_data = copy.deepcopy(data)
    indices = sorted(indices)
    split_data['annotations'] = [data['annotations'][i] for i in indices]
    split_data['images'] = []

    new_img_id = 0
    cur_img = -1
    for i, anno in enumerate(split_data['annotations']):
        img_id = anno['image_id']
        if cur_img != img_id:
            cur_img = img_id
            split_data['images'].append(data['images'][img_id])
            split_data['images'][-1]['id'] = new_img_id
            new_img_id += 1
        # reassign IDs
        anno['id'] = i
        anno['image_id'] = split_data['images'][-1]['id']

    with open(output_name, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"Data split saved to {output_name}")

def main():
    master_json_path = '../data/cropped_merged_annotations.json'
    with open(master_json_path, 'r') as f:
        data = json.load(f)

    test_videos = ['BP_2021-06-01_15-05-46_096578_0000000', 
                   'BP_2022-09-07_12-33-47_959910_0000000', 
                   'BP_2023-06-23_14-44-16_556681_0380000']

    # for each annotation, group by color and then by video.
    # color_groups[color][video_name] = list of annotation indices.
    color_groups = {}
    for anno in data['annotations']:
        color = get_effective_color(anno, data['images'])
        img = data['images'][anno['image_id']]
        # extract video name from file path (2nd to last elem)
        parts = img['file_name'].split('/')
        if len(parts) < 2:
            continue
        video_name = parts[-2]
        # init nested dictionnary 
        if color not in color_groups:
            color_groups[color] = {}
        if video_name not in color_groups[color]:
            color_groups[color][video_name] = []
        color_groups[color][video_name].append(anno['id'])

    train_ids_all = []
    val_ids_all = []
    test_ids_all = []

    for color, videos in color_groups.items():
        video_names = list(videos.keys())
        train_val_videos = [vn for vn in video_names if vn not in test_videos]
        test_videos_color = [vn for vn in video_names if vn in test_videos]
        
        train_vids, val_vids, test_vids_split = split_list(train_val_videos, [0.7, 0.2, 0.1])
        test_vids = test_vids_split + test_videos_color
        
        for vn in train_vids:
            train_ids_all.extend(videos[vn])
        for vn in val_vids:
            val_ids_all.extend(videos[vn])
        for vn in test_vids:
            test_ids_all.extend(videos[vn])

    random.shuffle(train_ids_all)
    random.shuffle(val_ids_all)
    random.shuffle(test_ids_all)

    save_split(data, train_ids_all, 'data/newdata_cls_train_vidsplit_n.json')
    save_split(data, val_ids_all, 'data/newdata_cls_val_vidsplit_n.json')
    save_split(data, test_ids_all, 'data/newdata_test_vidsplit_n.json')

if __name__ == "__main__":
    main()
