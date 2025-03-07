import numpy as np
from torch.utils.data import WeightedRandomSampler
import pdb
import re 

def extract_identity_number(identity_str):
    """
    Extract the first integer from an identity string.
    e.g. "bird_y_1" or "bird_b_2" returns 1 or 2, respectively. (see BP10_01 annotations)
    """
    if identity_str is None:
        raise ValueError("Received a None identity")
    match = re.search(r'\d+', identity_str)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No integer found in identity string: {identity_str}")


class ClassBalancedSampler:
    def __init__(self, labels):
        """
        Initializes the ClassBalancedSampler with the labels of the dataset.
        
        Args:
            labels (list or np.ndarray): The labels for the dataset.
        """
        self.labels = [extract_identity_number(i['identity']) for i in labels] # Convert each string to an integer
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
