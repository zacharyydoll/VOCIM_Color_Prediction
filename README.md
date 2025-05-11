# VOCIM Color Prediction

This repository contains the implementation of a color prediction system for birds in the VOCIM dataset. The system uses a combination of TinyViT for initial color predictions and a Graph Neural Network (GNN) for refining these predictions while ensuring bijectivity within each frame.
Based on work by Xiaoran Chen (SDSC).

## Architecture

The system consists of two main components:

1. **TinyViT Backbone**:
   - Processes individual bird crops
   - Outputs probability distribution over colors
   - Can use heatmap mask for better feature extraction

2. **ColorGNN**:
   - Takes TinyViT's predictions for birds in the same frame
   - Creates a bipartite graph between birds and colors
   - Refines predictions while ensuring bijectivity
   - Uses Hungarian algorithm for optimal assignments

## Workflow

1. **Data Processing**:
   - Images are cropped to individual birds
   - Frame IDs are extracted from filenames
   - Birds are grouped by frame for GNN processing

2. **Model Processing**:
   - TinyViT processes each crop individually
   - For each frame:
     a. Get top-K colors from TinyViT for each bird
     b. Create bipartite graph using only top-K colors
     c. GNN processes the graph to learn relationships
     d. Combine GNN scores with TinyViT probabilities
     e. Apply Hungarian algorithm for final assignments

3. **Score Combination**:
   - GNN outputs a matrix of shape (num_birds, num_colors)
   - Scores are weighted by TinyViT probabilities
   - Higher TinyViT confidence → stronger GNN influence
   - Lower TinyViT confidence → weaker GNN influence

4. **Bijectivity Constraint**:
   - Hungarian algorithm ensures one-to-one assignments
   - Each bird gets a unique color
   - Each color is used at most once per frame

## Key Features

- **Frame-Based Processing**: Birds from the same frame are processed together
- **Top-K Selection**: Only considers TinyViT's top-K color predictions
- **Bipartite Graph**: Represents relationships between birds and colors
- **Score Weighting**: GNN scores are weighted by TinyViT confidence
- **Bijective Assignments**: Ensures unique color assignments per frame

## Usage

1. **Training**:
```bash
python train.py --config config.py
```

2. **Evaluation**:
```bash
python eval.py --model_path path/to/model --data_path path/to/data
```

## Requirements

- PyTorch
- torch-geometric
- timm
- numpy
- PIL
- yaml

## Configuration

Key parameters in `config.py`:
- `use_heatmap_mask`: Whether to use heatmap mask for TinyViT
- `sigma_val`: Sigma value for heatmap mask
- Model architecture parameters
- Training parameters

## Dataset Structure

The dataset should be organized as follows:
- Images are cropped to individual birds
- Filenames contain frame IDs (e.g., 'img00332_bird_1.png')
- YAML files map bird identities to colors
- JSON annotations contain bounding boxes and metadata

## Notes

- The heatmap mask only affects TinyViT's feature extraction
- GNN processes TinyViT's outputs, not the original images
- Bijectivity is enforced at the frame level
- The system can handle varying numbers of birds per frame

## Project Overview

The system is designed to:
- Process bird images with backpack annotations
- Apply heatmap masks to focus on relevant regions
- Train a deep learning model to predict color categories
- Evaluate model performance on test datasets
- Enforce unique color assignments per frame using linear assignment

## Project Structure

- `model.py`: Contains the main model architecture and training logic
- `dataset.py`: Handles data loading and preprocessing
- `dataloader.py`: Manages data batching and loading
- `train.py`: Main training script
- `eval.py`: Evaluation script
- `linear_assignment_eval.py`: Evaluation with linear assignment for unique color constraints
- `utils.py`: Utility functions
- `config.py`: Configuration file containing global parameters and settings for the project, including:
  - Heatmap mask parameters (sigma values)
  - Model configuration settings
  - Data processing parameters
- `split.py`, `split_by_color_video.py`, `split_by_video.py`: Data splitting utilities
- `sampler.py`: Custom data sampling implementation
- `utils_and_analysis/`: Additional utility scripts and analysis tools

## Data Preparation

The project expects data in the following format:
- JSON annotations file containing image paths and labels
- YAML files for bird identity mapping (`newdata_bird_identity.yaml`)
- YAML file for color mapping (`newdata_colormap.yaml`)

Use the provided splitting scripts to prepare your dataset:
- `split.py`: General data splitting
- `split_by_color_video.py`: Split data by color and video
- `split_by_video.py`: Split data by video
- `split_by_videosplit_file.py`: Split data using a predefined split file

## Training

To train the model:

1. Prepare your dataset using the splitting scripts
2. Run the training script:
```bash
./train.sh
```

The training script supports:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Logging of training metrics

## Evaluation

To evaluate a trained model:
```bash
./eval.sh
```

The evaluation script will:
- Load the trained model
- Run inference on the test set
- Calculate accuracy and other metrics
- Save predictions to a JSON file

For evaluation with linear assignment:
```bash
python linear_assignment_eval.py
```

The linear assignment evaluation:
- Groups predictions by frame
- Creates cost matrices using model probabilities
- Applies the Hungarian algorithm to enforce unique color assignments
- Reports both direct and linear assignment accuracies

## Model Architecture

The model uses a deep learning architecture that:
- Takes RGB images as input
- Optionally uses heatmap masks to focus on relevant regions
- Processes images through a neural network
- Outputs color category predictions

### GNN Enhancement

The model includes a Graph Neural Network (GNN) component that:
- Enhances the base TinyViT architecture
- Uses separate dropout for the GNN component
- Creates a bipartite graph between birds and colors
- Processes relationships between birds and colors in the same frame
- Currently achieves ~92.8% accuracy on the ambiguous subset

## Linear Assignment Evaluation

The linear assignment evaluation enforces the constraint that each color can only be used once per frame:
1. Groups all birds from the same frame together
2. Creates a cost matrix for each frame where:
   - Rows represent bird predictions
   - Columns represent available colors
   - Cell values are 1 - model's softmax probabilities
3. Uses the Hungarian algorithm to find optimal unique color assignments
4. Computes accuracy across all frames

Results show that enforcing unique color assignments improves accuracy:
- Normal test set: 96.64% → 98.03%
- Ambiguous test set: 92.81% → 94.96%

## Contact

zachary.doll@epfl.ch
