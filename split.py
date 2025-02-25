from sklearn.model_selection import train_test_split
import json
import pandas as pd
from math import isclose
import copy

def stratified_split(data, labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=None):
    """
    Perform a stratified split for classification to ensure that each label
    has the same proportion in train, validation, and test sets.

    Parameters:
    - data: array-like or DataFrame, features or data to split.
    - labels: array-like, target labels corresponding to the data.
    - train_ratio: float, proportion of data for the training set.
    - val_ratio: float, proportion of data for the validation set.
    - test_ratio: float, proportion of data for the test set.
    - random_state: int, random state for reproducibility.

    Returns:
    - X_train, X_val, X_test: split data.
    - y_train, y_val, y_test: split labels.
    """
    # Ensure the ratios sum to 1
    assert isclose(train_ratio + val_ratio + test_ratio, 1.0) == 1.0, "Train, val, and test ratios must sum to 1."

    # First split: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, labels, test_size=test_ratio, stratify=labels, random_state=random_state
    )

    # Compute the proportion of validation data relative to the train+val set
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)

    # Second split: train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_ratio, stratify=y_train_val, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Example usage:
if __name__ == "__main__":
    f = open('/mydata/vocim/xiaoran/scripts/bird_identity_classification/data/newdata_cls_trainvaltest.json','r')
    cls_data = json.load(f)
    data = [entry['id'] for entry in cls_data['annotations']]
    labels = [entry['identity'] for entry in cls_data['annotations']]

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        data, labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=44
    )

    print("Train set size:", len(X_train))
    print("Validation set size:", len(X_val))
    print("Test set size:", len(X_test))

    print("Label distribution in train set:", pd.Series(y_train).value_counts(normalize=True))
    print("Label distribution in validation set:", pd.Series(y_val).value_counts(normalize=True))
    print("Label distribution in test set:", pd.Series(y_test).value_counts(normalize=True))

    split_dict = dict()
    split_dict['train'] = X_train
    split_dict['val'] = X_val
    split_dict['test'] = X_test

    with open("data/data_split.json", "w") as json_file:
        json.dump(split_dict, json_file)
    print(f'data split saved to data/data_split.json')

    train_data = copy.deepcopy(cls_data)
    train_data['annotations'] = [cls_data['annotations'][i] for i in X_train]
    train_data['images'] = [cls_data['images'][i] for i in X_train] 

    for i, entry in enumerate(train_data['annotations']):
        entry['id'] = i
        entry['image_id'] = i
    for i, entry in enumerate(train_data['images']):
        entry['id'] = i

    with open("data/newdata_cls_train.json", "w") as json_file:
        json.dump(train_data, json_file)
        print(f'data split saved to data/newdata_cls_train.json')


    val_data = copy.deepcopy(cls_data)
    val_data['annotations'] = [cls_data['annotations'][i] for i in X_val]
    val_data['images'] = [cls_data['images'][i] for i in X_val]

    for i, entry in enumerate(val_data['annotations']):
        entry['id'] = i
        entry['image_id'] = i
    for i, entry in enumerate(val_data['images']):
        entry['id'] = i

    with open("data/newdata_cls_val.json", "w") as json_file:
        json.dump(val_data, json_file)
        print(f'data split saved to data/newdata_cls_val.json')

    test_data = copy.deepcopy(cls_data)
    test_data['annotations'] = [cls_data['annotations'][i] for i in X_test]
    test_data['images'] = [cls_data['images'][i] for i in X_test]    
    
    for i, entry in enumerate(test_data['annotations']):
        entry['id'] = i
        entry['image_id'] = i
    for i, entry in enumerate(test_data['images']):
        entry['id'] = i

    with open("data/newdata_cls_test.json", "w") as json_file:
        json.dump(test_data, json_file)
        print(f'data split saved to data/newdata_cls_test.json')


    