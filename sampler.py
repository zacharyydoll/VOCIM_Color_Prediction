import numpy as np
from torch.utils.data import WeightedRandomSampler
import pdb
import re 

class ClassBalancedSampler:
    def __init__(self, dataset):
        """
        Initializes the ClassBalancedSampler with the labels of the dataset.
        
        Args:
            labels (list or np.ndarray): The labels for the dataset.
        """
        self.dataset = dataset
        self.labels = [self.dataset.get_effective_label(anno) for anno in self.dataset.annotations]
        self.class_weights = self._compute_class_weights()
        self.sample_weights = self._compute_sample_weights()
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
