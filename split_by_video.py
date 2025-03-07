import json
import copy
import random
import math

def get_video_groups(data):
    video_groups = {}

    for idx, image in enumerate(data['images']):
        video_name = image['file_name'].split('/')[-2]
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(idx)
    return video_groups

def split_list(items, ratio):
    random.seed(42)
    items = items[:]  # makes copy
    random.shuffle(items)

    n = len(items)
    train_n = int(n * ratio[0])
    val_n = int(n * ratio[1])

    train = items[:train_n]
    val = items[train_n:train_n+val_n]
    test = items[train_n+val_n:]

    return train, val, test

def save_split(data, indices, output_name):
    split_data = copy.deepcopy(data)
    indices = sorted(indices)
    split_data['annotations'] = [data['annotations'][i] for i in indices]
    split_data['images'] = []

    new_img_id = 0
    cur_id = -1

    for i, entry in enumerate(split_data['annotations']):
        img_id = entry['image_id']
        if cur_id == -1 or img_id != cur_id:
            cur_id = img_id
            split_data['images'].append(data['images'][img_id])
            split_data['images'][-1]['id'] = new_img_id
            new_img_id += 1
        entry['id'] = i
        entry['image_id'] = split_data['images'][-1]['id']

    with open(output_name, "w") as json_file:
        json.dump(split_data, json_file)
        print(f'data split saved to {output_name}')

def main():
    with open('/mydata/vocim/zachary/data/cropped_merged_annotations.json', 'r') as f:
        data = json.load(f)
    
    video_groups = get_video_groups(data) # Get video grps directly from master json instead of CSV
    video_names = list(video_groups.keys())
    train_videos, val_videos, test_videos = split_list(video_names, [0.7, 0.2, 0.1])

    # Collect annotations idx for each group 
    train_ids = [idx for video in train_videos for idx in video_groups[video]]
    val_ids = [idx for video in val_videos for idx in video_groups[video]]
    test_ids = [idx for video in test_videos for idx in video_groups[video]]

    save_split(data, train_ids, 'data/vocim_yolopose_train_vidsplit.json')
    save_split(data, val_ids, 'data/vocim_yolopose_val_vidsplit.json')
    save_split(data, test_ids, 'data/vocim_yolopose_test_vidsplit.json')

if __name__ == "__main__":
    main()
