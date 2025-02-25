import json
from split_by_video import save_split

view = 'sideview'
with open('video_split.json', 'r') as f:
    splits = json.load(f)

train_videos = set(splits['train'])
val_videos = set(splits['val'])
test_videos = set(splits['test'])

with open(f'/mydata/vocim/xiaoran/scripts/multiview_video_keypoint_predict/vocim_yolopose_trainvaltest_{view}.json', 'r') as f:
    data = json.load(f)

test_videos = ['BP_2021-06-01_15-05-46_096578_0000000', 'BP_2022-09-07_12-33-47_959910_0000000', 'BP_2023-06-23_14-44-16_556681_0380000']

train_idx = []
val_idx = []
test_idx = []

for image, anno in zip(data['images'], data['annotations']):
    idx = anno['idx']
    video_name = image['file_name'].split('/')[-2]
    if video_name in train_videos:
        train_idx.append(idx)
    elif video_name in val_videos:
        val_idx.append(idx)
    elif video_name in test_videos:
        test_idx.append(idx)
    else:
        train_idx.append(idx)

save_split(data, train_idx, f'data/vocim_yolopose_train_vidsplit_{view}.json')
save_split(data, val_idx, f'data/vocim_yolopose_val_vidsplit_{view}.json')
save_split(data, test_idx, f'data/vocim_yolopose_test_vidsplit_{view}.json')


