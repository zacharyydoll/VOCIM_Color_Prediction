# VOCIM Color Prediction

This project is a deep learning system for color prediction in bird images, specifically focusing on identifying the color of backpacks worn by birds. The system uses a combination of computer vision and deep learning techniques to analyze images and predict color categories.
Based on work by Xiaoran Chen (SDSC).

## Project Overview

The system is designed to:
- Process bird images with backpack annotations
- Apply heatmap masks to focus on relevant regions
- Train a deep learning model to predict color categories
- Evaluate model performance on test datasets

## Requirements

The project requires Python 3.9 and the following key dependencies:
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 12.1
- Other dependencies listed in `environment.yml`

## Project Structure

- `model.py`: Contains the main model architecture and training logic
- `dataset.py`: Handles data loading and preprocessing
- `dataloader.py`: Manages data batching and loading
- `train.py`: Main training script
- `eval.py`: Evaluation script
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

## Model Architecture

The model uses a deep learning architecture that:
- Takes RGB images as input
- Optionally uses heatmap masks to focus on relevant regions
- Processes images through a neural network
- Outputs color category predictions

## Contact

zachary.doll@epfl.ch
