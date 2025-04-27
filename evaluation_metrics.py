import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_networkx
import networkx as nx
from config import (
    compute_confusion_matrix, compute_class_metrics,
    compute_roc_auc, compute_f1_score, compute_precision_recall,
    compute_graph_metrics
)
import pickle
import os

EVAL_RESULTS_DIR = 'eval_results'
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

class ModelEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.graph_data = []  # Store graph data for visualization
        self.edge_weights = []  # Store edge weights for analysis

    def update(self, outputs, labels, graph_data=None):
        # Convert outputs to probabilities if they're logits
        if not torch.is_tensor(outputs):
            outputs = torch.tensor(outputs)
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            probs = torch.softmax(outputs, dim=1)
        else:
            probs = torch.sigmoid(outputs)
        
        # Get predictions
        preds = torch.argmax(probs, dim=1)
        
        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
        
        if graph_data is not None:
            self.graph_data.append(graph_data)
            # Store edge weights for analysis
            if hasattr(graph_data, 'edge_attr'):
                self.edge_weights.extend(graph_data.edge_attr.cpu().numpy())

    def compute_metrics(self):
        metrics = {}
        
        if compute_confusion_matrix:
            cm = confusion_matrix(self.labels, self.predictions)
            metrics['confusion_matrix'] = cm
            self._plot_confusion_matrix(cm)

        if compute_class_metrics:
            report = classification_report(self.labels, self.predictions, output_dict=True)
            metrics['classification_report'] = report

        if compute_roc_auc:
            try:
                roc_auc = roc_auc_score(self.labels, self.probabilities, multi_class='ovr')
                metrics['roc_auc'] = roc_auc
            except:
                metrics['roc_auc'] = None

        if compute_f1_score:
            f1 = f1_score(self.labels, self.predictions, average='weighted')
            metrics['f1_score'] = f1

        if compute_precision_recall:
            avg_precision = average_precision_score(self.labels, self.probabilities)
            metrics['average_precision'] = avg_precision
            self._plot_precision_recall_curve()

        # Add graph-specific metrics and visualizations
        if compute_graph_metrics and self.graph_data:
            self._visualize_graph_structure()
            metrics['graph_metrics'] = self._compute_graph_metrics()
            if self.edge_weights:
                self._plot_edge_weight_distribution()
                metrics['edge_metrics'] = self._compute_edge_metrics()

        accuracy = accuracy_score(self.labels, self.predictions)
        metrics['accuracy'] = accuracy

        return metrics

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'confusion_matrix.png'))
        plt.close()

    def _plot_precision_recall_curve(self):
        plt.figure(figsize=(10, 6))
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(
                np.array(self.labels) == i,
                np.array(self.probabilities)[:, i]
            )
            plt.plot(recall, precision, label=f'Class {i}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'precision_recall_curve.png'))
        plt.close()

    def _visualize_graph_structure(self):
        if not self.graph_data:
            return
            
        # Take a sample graph for visualization
        sample_graph = self.graph_data[0]
        G = to_networkx(sample_graph, to_undirected=True)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=500)
        
        # Draw edges with weights
        if hasattr(sample_graph, 'edge_attr'):
            edge_weights = sample_graph.edge_attr.cpu().numpy()
            nx.draw_networkx_edges(G, pos, width=edge_weights)
        else:
            nx.draw_networkx_edges(G, pos)
            
        plt.title('Graph Structure Visualization')
        plt.savefig('graph_structure.png')
        plt.close()
        
    def _plot_edge_weight_distribution(self):
        if not self.edge_weights:
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(self.edge_weights, bins=50)
        plt.title('Edge Weight Distribution')
        plt.xlabel('Edge Weight')
        plt.ylabel('Frequency')
        plt.savefig('edge_weight_distribution.png')
        plt.close()
        
    def _compute_graph_metrics(self):
        graph_metrics = {}
        
        # Compute average node degree
        total_degree = 0
        total_nodes = 0
        for graph in self.graph_data:
            G = to_networkx(graph, to_undirected=True)
            total_degree += sum(dict(G.degree()).values())
            total_nodes += G.number_of_nodes()
            
        graph_metrics['average_degree'] = total_degree / total_nodes if total_nodes > 0 else 0
        
        # Compute graph density
        total_edges = 0
        for graph in self.graph_data:
            G = to_networkx(graph, to_undirected=True)
            total_edges += G.number_of_edges()
            
        graph_metrics['average_density'] = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        
        # Compute clustering coefficient
        total_clustering = 0
        for graph in self.graph_data:
            G = to_networkx(graph, to_undirected=True)
            total_clustering += nx.average_clustering(G)
            
        graph_metrics['average_clustering'] = total_clustering / len(self.graph_data) if self.graph_data else 0
        
        return graph_metrics
        
    def _compute_edge_metrics(self):
        edge_metrics = {}
        
        if self.edge_weights:
            edge_weights = np.array(self.edge_weights)
            edge_metrics['mean_edge_weight'] = np.mean(edge_weights)
            edge_metrics['std_edge_weight'] = np.std(edge_weights)
            edge_metrics['min_edge_weight'] = np.min(edge_weights)
            edge_metrics['max_edge_weight'] = np.max(edge_weights)
            
        return edge_metrics

    def reset(self):
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.graph_data = []
        self.edge_weights = []

if __name__ == '__main__':
    import pickle
    import os
    EVAL_RESULTS_DIR = 'eval_results'
    pkl_path = os.path.join(EVAL_RESULTS_DIR, 'evaluation_metrics.pkl')
    try:
        with open(pkl_path, 'rb') as f:
            metrics = pickle.load(f)

        log_path = os.path.join(EVAL_RESULTS_DIR, 'evaluation_metrics.log')
        with open(log_path, 'w') as log_file:
            for key, value in metrics.items():
                log_file.write(f"{key}: {value}\n\n")
        print(f"All metrics saved to {log_path}")
    except FileNotFoundError:
        print(f"{pkl_path} not found. Please run eval.py first to generate it.")