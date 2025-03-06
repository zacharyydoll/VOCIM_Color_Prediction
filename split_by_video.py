import json
import os
import copy
import random
import math

def summarize_data(data_path):
    import glob
    import pandas as pd

    annotation_folders = glob.glob(os.path.join(data_path, "*BP*/labeled*/BP*"))
    summary = dict()
    for folder in annotation_folders:
        df_filename = [os.path.join(data_path, folder, i) for i in os.listdir(folder) if i.endswith('.csv')]
        if len(df_filename):
            df_filename = df_filename[0]
        else:
            df_filename = None
            print(f"{folder} has no annotations")
                
        img_filenames = [os.path.join(folder,i) for i in os.listdir(folder) if i.endswith('.png') or i.endswith('.jpg')]
        if not len(img_filenames):
            print(f"{folder} has no images" )

        if df_filename:
            df = pd.read_csv(df_filename, header=[0,1,2,3])
            ncols = len(df.columns)-3
            nbird = ncols/10
            if nbird not in summary.keys():
                summary[nbird]=[[], []]
            
            summary[nbird][0].append(folder)
            summary[nbird][1].append(img_filenames)
    return summary


def split(indices, ratio):

    assert len(ratio)==3, "set split ratio such as [0.7, 0.2, 0.1]" 
    train_ratio, val_ratio, test_ratio = ratio
    assert train_ratio<1 and val_ratio<1 and test_ratio<1 and (train_ratio+val_ratio+test_ratio)>0.9999, "ratios should be lower than 1 and sum up to 1"

    random.seed(42)

    random.shuffle(indices)
    len_indices = len(indices)

    if len_indices<3:
        out = [[indices[i]] if i<len_indices else [] for i in range(3)]
        return 
    if len_indices==3:
        return [[indices[i]] for i in range(3)]

    intervals = [math.floor(len_indices * train_ratio), math.ceil(len_indices * val_ratio)]
    intervals = [int(i) for i in intervals]

    train_indices = indices[:intervals[0]]
    val_indices = indices[intervals[0]:intervals[0]+intervals[1]]
    test_indices = indices[intervals[0]+intervals[1]:]

    if not len(test_indices) and len(val_indices)>1:
        test_indices = [val_indices[-1]]
        val_indices = val_indices[:-1]
    return [train_indices, val_indices, test_indices] 

def save_split(data, indices, output_name):
    split_data = copy.deepcopy(data)
    indices = sorted(indices)
    
    split_data['annotations'] = [data['annotations'][i] for i in indices]
    data_ids = [data['annotations'][i]['id'] for i in indices]
    split_data['images'] = []
    # split_data['images'] = [data['images'][i] for i in data_ids] 

    new_img_id = 0
    cur_id = -1
    for i, entry in enumerate(split_data['annotations']):        
        data_id = entry['id']
        img_id = entry['image_id']

        if cur_id ==-1 or img_id!=cur_id:
            cur_id = img_id

            split_data['images'].append(data['images'][img_id])
            split_data['images'][-1]['id'] = new_img_id
            new_img_id+=1

        entry['id'] = i
        entry['image_id'] = split_data['images'][-1]['id']

    with open(output_name, "w") as json_file:
        json.dump(split_data, json_file)
        print(f'data split saved to {output_name}')

def main():
    with open('/mydata/vocim/xiaoran/scripts/multiview_video_keypoint_predict/vocim_yolopose_trainvaltest.json', 'r') as f:
        data = json.load(f)

    test_videos = ['BP_2021-06-01_15-05-46_096578_0000000', 'BP_2022-09-07_12-33-47_959910_0000000', 'BP_2023-06-23_14-44-16_556681_0380000']
    data_by_n = dict()

    train_ids_all_n = []
    val_ids_all_n = []
    test_ids_all_n = []

    train_video_list = []
    val_video_list = []
    test_video_list = []

    data_by_n = dict()
    summary = summarize_data('/mydata/vocim/shared/KeypointAnnotations')

    for k in summary.keys():
        for v in summary[k][0]:
            _v = v.split('/')[-1]
            data_by_n[_v] = k
    
    for k in summary.keys():
        data_summary = dict()
        
        for image, anno in zip(data['images'], data['annotations']):
            video_name = image['file_name'].split('/')[-2]
            if data_by_n[video_name]!=k or video_name in test_videos:
                continue

            idx = anno['id']
            year = video_name[3:7]

            if year not in data_summary.keys():
                data_summary[year] = dict()
            if video_name not in data_summary[year].keys():
                data_summary[year][video_name] = []
            
            data_summary[year][video_name].append(idx)

        # for image, anno in zip(data['images'], data['annotations']):
        #     video_name = image['file_name'].split('/')[-2]
        #     if data_by_n[video_name]!=k or video_name in test_videos:
        #         continue

        #     color_label = anno['identity']
        #     idx = anno['id']
            
        #     if color_label not in data_summary.keys():
        #         data_summary[color_label] = dict()
            
        #     if video_name not in data_summary[color_label].keys():
        #         data_summary[color_label][video_name] = []

        #     data_summary[color_label][video_name].append(idx)

        train_ids = []
        val_ids = []
        test_ids = []

        for year in data_summary.keys():
            vids = list(data_summary[year].keys())
            vid_indices = list(range(len(vids)))
            train, val, test = split(vid_indices, [0.7, 0.2, 0.1])
            # import pdb
            # pdb.set_trace()
            train_ids.extend([data_summary[year][vids[i]] for i in train])
            val_ids.extend([data_summary[year][vids[i]] for i in val])
            test_ids.extend([data_summary[year][vids[i]] for i in test])

            train_vids = [vids[i] for i in train]
            val_vids = [vids[i] for i in val]
            test_vids = [vids[i] for i in test]

            train_video_list.extend(train_vids)
            val_video_list.extend(val_vids)
            test_video_list.extend(test_vids)

        train_ids = [i for ids in train_ids for i in ids]
        val_ids = [i for ids in val_ids for i in ids]
        test_ids = [i for ids in test_ids for i in ids]

        train_ids_all_n.append(train_ids)
        val_ids_all_n.append(val_ids)
        test_ids_all_n.append(test_ids)

    video_split = {'train': train_video_list,
                    'val': val_video_list,
                    'test': test_video_list}

    with open('video_split.json', 'w') as f:
        json.dump(video_split, f)

    train_ids_all_n = [i for ids in train_ids_all_n for i in ids]
    val_ids_all_n = [i for ids in val_ids_all_n for i in ids]
    test_ids_all_n = [i for ids in test_ids_all_n for i in ids]

    save_split(data, train_ids_all_n, 'data/vocim_yolopose_train_vidsplit.json')
    save_split(data, val_ids_all_n, 'data/vocim_yolopose_val_vidsplit.json')
    save_split(data, test_ids_all_n, 'data/vocim_yolopose_test_vidsplit.json')


main()