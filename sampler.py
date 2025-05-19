import numpy as np
from torch.utils.data import WeightedRandomSampler
import pdb
import re 
import json
import os

from config import weigh_ambig_samples

class ClassBalancedSampler:
    def __init__(self, dataset, ambiguous_json_path=None, ambiguous_factor=5):
        """
        Initializes the ClassBalancedSampler with the labels of the dataset.
        Optionally oversamples ambiguous cases.
        Args:
            dataset: The dataset instance.
            ambiguous_json_path (str): Path to JSON file with ambiguous file names.
            ambiguous_factor (int): Factor to multiply ambiguous sample weights.
        """
        self.dataset = dataset
        self.labels = [self.dataset.get_effective_label(anno) for anno in self.dataset.annotations]
        self.class_weights = self._compute_class_weights()
        self.sample_weights = self._compute_sample_weights()

        # Oversample ambiguous cases if provided
        if weigh_ambig_samples and ambiguous_json_path and os.path.exists(ambiguous_json_path):
            with open(ambiguous_json_path, 'r') as f:
                ambiguous_files = set(json.load(f))
            for idx, annotation in enumerate(self.dataset.annotations):
                img_idx = annotation['image_id']
                file_name = self.dataset.img_paths[img_idx]['file_name']
                if file_name in ambiguous_files:
                    self.sample_weights[idx] *= ambiguous_factor

        self.sampler = WeightedRandomSampler(weights=self.sample_weights, 
                                             num_samples=len(self.sample_weights), 
                                             replacement=True)

    def _compute_class_weights(self):
        """
        Computes the class weights as the inverse of class frequencies.
        
        Returns:
            np.ndarray: Array of class weights.
        """
        epsilon = 1e-8 # just to avoid division by 0 
        class_counts = np.bincount(self.labels)
        total_count = len(self.labels)
        # Compute weights as the inverse of class frequencies
        class_weights = total_count / (len(class_counts) * (class_counts + epsilon)) # might need to take out the epsilon value
        return class_weights

    def _compute_sample_weights(self):
        """
        Computes the sample weights based on class weights.
        
        Returns:
            np.ndarray: Array of sample weights.
        """
        return self.class_weights[self.labels]

    def get_sampler(self):
        """
        Returns the WeightedRandomSampler instance.
        
        Returns:
            WeightedRandomSampler: The sampler for the DataLoader.
        """
        return self.sampler
