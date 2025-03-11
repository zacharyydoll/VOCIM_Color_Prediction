import pickle
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
    cm = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions)
    return acc, cm, report

def visualize_misclassified(image_paths, labels, predictions, num_images=5):
    """
    Visualize a few misclassified images.
    """
    # Identify indices where the prediction does not match the true label
    misclassified_indices = [i for i, (pred, true) in enumerate(zip(predictions, labels)) if pred != true]
    print("Number of misclassified samples:", len(misclassified_indices))
    
    # Display the first 'num_images' misclassified samples
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
        plt.show()

def save_classification_report(labels, predictions, filename="classification_report.json"):
    """
    Save the classification report as a JSON file.
    """
    report_dict = classification_report(labels, predictions, output_dict=True)
    with open(filename, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to {filename}")

def main():
    # Define the results filename (assumed to be in the working directory)
    results_file = "output_top.pkl"
    
    # Load the results dictionary
    results = load_results(results_file)
    
    # Extract predictions, true labels, and image paths
    predictions = results.get("predictions")
    labels = results.get("labels")
    image_paths = results.get("image_paths")
    
    if predictions is None or labels is None:
        print("Error: Predictions or labels not found in the results file.")
        return

    # Compute evaluation metrics
    accuracy, cm, report = compute_metrics(labels, predictions)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    
    # Save the classification report to a JSON file
    save_classification_report(labels, predictions, filename="classification_report.json")
    
    # Visualize a few misclassified examples for error analysis
    visualize_misclassified(image_paths, labels, predictions, num_images=5)

if __name__ == "__main__":
    main()
