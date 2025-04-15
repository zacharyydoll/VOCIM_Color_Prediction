import pickle
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_results(filename):
    """
    Load the prediction results from a pickle file.
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results

def compute_metrics(labels, predictions):
    """
    Compute evaluation metrics: accuracy, confusion matrix, and classification report.
    """
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions).tolist() 
    report = classification_report(labels, predictions, output_dict=True)
    return acc, cm, report

def visualize_misclassified(image_paths, labels, predictions, num_images=5):
    """
    Visualize and save a few misclassified images.
    """
    misclassified_indices = [i for i, (pred, true) in enumerate(zip(predictions, labels)) if pred != true]
    print("Number of misclassified samples:", len(misclassified_indices))
    
    # OP directory for misclassified images
    output_dir = "/mydata/vocim/zachary/color_prediction/mask_comparisons/resnet50/missclassfied_without_msk"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the first num_images misclassified examples
    for idx in misclassified_indices[:num_images]:
        img_path = image_paths[idx]
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        plt.imshow(img)
        plt.title(f"True: {labels[idx]}  |  Predicted: {predictions[idx]}")
        plt.axis('off')
        save_path = os.path.join(output_dir, f"missclassified_{idx}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved misclassified image to {save_path}")

def save_classification_report(labels, predictions, output_path):
    """
    Save the classification metrics as a JSON file.
    """
    acc, cm, report = compute_metrics(labels, predictions)
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Classification report saved to {output_path}")

def main():
    # results file name from eval.py
    #results_file = "../output_top.pkl"
    results_file = "/mydata/vocim/zachary/color_prediction/mask_comparisons/resnet50/resnet_no_mask_ambig_set.pkl"
    
    results = load_results(results_file)
    
    # extract predictions, true labels and image paths
    predictions = results.get("predictions")
    labels = results.get("labels")
    image_paths = results.get("image_paths")
    
    if predictions is None or labels is None:
        print("Error: Predictions or labels not found in the results file.")
        return

    acc, cm, report = compute_metrics(labels, predictions)
    print("Accuracy:", acc)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    
    output_path = "/mydata/vocim/zachary/color_prediction/mask_comparisons/resnet50/resnet_report_without_mask.json"
    save_classification_report(labels, predictions, output_path)
    
    # Visualize a few misclassified examples for error analysis
    visualize_misclassified(image_paths, labels, predictions, num_images=10)

if __name__=="__main__":
    main()
