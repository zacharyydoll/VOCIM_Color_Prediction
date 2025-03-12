import json
import os
import copy
import random
import math

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
    # Ensure that each split gets at least one item if possible
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
            # Append this image only once.
            split_data['images'].append(data['images'][img_id])
            split_data['images'][-1]['id'] = new_img_id
            new_img_id += 1
        # Reassign IDs
        anno['id'] = i
        # Set the annotation image_id to the new image id
        anno['image_id'] = split_data['images'][-1]['id']

    with open(output_name, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"Data split saved to {output_name}")

def main():
    # Load the master JSON (adjust the path as needed)
    master_json_path = '../data/cropped_merged_annotations.json'
    with open(master_json_path, 'r') as f:
        data = json.load(f)

    # Define a list of videos that you want to assign to test set regardless of color
    test_videos = ['BP_2021-06-01_15-05-46_096578_0000000', 
                   'BP_2022-09-07_12-33-47_959910_0000000', 
                   'BP_2023-06-23_14-44-16_556681_0380000']

    # For each annotation, group by color and then by video.
    # We'll build a dictionary:
    # color_groups[color][video_name] = list of annotation indices.
    color_groups = {}
    for anno in data['annotations']:
        color = anno['identity']  # your backpack color label, e.g., "bird_1"
        img = data['images'][anno['image_id']]
        # Extract video name from file path; assume video folder is second-to-last element.
        parts = img['file_name'].split('/')
        if len(parts) < 2:
            continue
        video_name = parts[-2]
        # Initialize nested dictionary
        if color not in color_groups:
            color_groups[color] = {}
        if video_name not in color_groups[color]:
            color_groups[color][video_name] = []
        color_groups[color][video_name].append(anno['id'])

    # Now, for each color, split the videos into train/val/test.
    train_ids_all = []
    val_ids_all = []
    test_ids_all = []

    for color, videos in color_groups.items():
        # Get list of video names for this color.
        video_names = list(videos.keys())
        # Exclude videos that are predetermined for testing.
        train_val_videos = [vn for vn in video_names if vn not in test_videos]
        test_videos_color = [vn for vn in video_names if vn in test_videos]
        
        # Split the train_val_videos using your desired ratio.
        train_vids, val_vids, test_vids_split = split_list(train_val_videos, [0.7, 0.2, 0.1])
        # Combine predetermined test videos with those from split.
        test_vids = test_vids_split + test_videos_color
        
        # For each set, collect the annotation indices.
        for vn in train_vids:
            train_ids_all.extend(videos[vn])
        for vn in val_vids:
            val_ids_all.extend(videos[vn])
        for vn in test_vids:
            test_ids_all.extend(videos[vn])

    # Optionally, shuffle the indices from each set
    random.shuffle(train_ids_all)
    random.shuffle(val_ids_all)
    random.shuffle(test_ids_all)

    # Save the splits.
    save_split(data, train_ids_all, 'data/newdata_cls_train_vidsplit_n.json')
    save_split(data, val_ids_all, 'data/newdata_cls_val_vidsplit_n.json')
    save_split(data, test_ids_all, 'data/newdata_test_vidsplit_n.json')

if __name__ == "__main__":
    main()
