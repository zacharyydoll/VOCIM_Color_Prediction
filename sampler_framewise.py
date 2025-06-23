from torch.utils.data import Sampler
import random
from collections import defaultdict, Counter
import numpy as np
import os
import json

class FrameBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, ambiguous_json_path=None, ambiguous_factor=5, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by frame_id
        self.frame_to_indices = defaultdict(list)
        self.frame_to_classes = defaultdict(list)
        self.frame_to_ambiguous = defaultdict(bool)
        for idx, annotation in enumerate(self.dataset.annotations):
            img_idx = annotation['image_id']
            file_name = self.dataset.img_paths[img_idx]['file_name']
            frame_id = file_name.split('_bird_')[0]
            self.frame_to_indices[frame_id].append(idx)
            # get crop's class label
            label = self.dataset.get_effective_label(annotation)
            self.frame_to_classes[frame_id].append(label)
        self.frames = list(self.frame_to_indices.keys())

        # assign single class to each frame (majority class or first crop's class)
        self.frame_class = {}
        for frame_id, class_list in self.frame_to_classes.items():
            self.frame_class[frame_id] = Counter(class_list).most_common(1)[0][0]

        # compute class weights (inverse freq)
        all_classes = [self.frame_class[fid] for fid in self.frames]
        class_counts = np.bincount(all_classes)
        total_count = len(all_classes)
        epsilon = 1e-8
        class_weights = total_count / (len(class_counts) * (class_counts + epsilon))
        self.frame_weights = {fid: class_weights[self.frame_class[fid]] for fid in self.frames}

        # Mark ambiguous frames
        ambiguous_files = set()
        if ambiguous_json_path and os.path.exists(ambiguous_json_path):
            with open(ambiguous_json_path, 'r') as f:
                ambiguous_files = set(json.load(f))
        for frame_id, indices in self.frame_to_indices.items():
            for idx in indices:
                img_idx = self.dataset.annotations[idx]['image_id']
                file_name = self.dataset.img_paths[img_idx]['file_name']
                if file_name in ambiguous_files:
                    self.frame_to_ambiguous[frame_id] = True
                    break

        # build sampling pool with oversampling and class balancing
        self.sampling_pool = []
        for frame_id in self.frames:
            weight = self.frame_weights[frame_id]
            n_copies = ambiguous_factor if self.frame_to_ambiguous[frame_id] else 1
            # multiply by weight 
            n_copies = int(np.round(n_copies * weight))
            n_copies = max(n_copies, 1)
            self.sampling_pool.extend([frame_id] * n_copies)

    def __iter__(self):
        pool = self.sampling_pool.copy()
        if self.shuffle:
            random.shuffle(pool)
        batch = []
        for frame_id in pool:
            indices = self.frame_to_indices[frame_id]
            if len(batch) + len(indices) > self.batch_size and batch:
                yield batch
                batch = []
            batch.extend(indices)
        if batch:
            yield batch

    def __len__(self):
        total = sum(len(self.frame_to_indices[fid]) for fid in self.sampling_pool)
        return (total + self.batch_size - 1) // self.batch_size