# USRA: Burnt Area Mapping Using Satellite Imagery Semantic Segmentation Research

This repository contains the implementation and trained models for semantic segmentation of satellite imagery using deep learning techniques. The project focuses on developing robust neural network architectures for pixel-wise classification of satellite imagery for burnt area mapping.

## Overview

The workflow consists of three main stages:

1. **Data Preprocessing** - Preparation and augmentation of satellite imagery datasets
2. **Model Training** - Training custom U-Net and GRU-based architectures
3. **Inference** - Applying trained models to segment new satellite images

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/bsvskashyap/usra.git
cd usra
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- TensorFlow 2.x / Keras
- NumPy
- OpenCV
- Matplotlib

## Usage

### 1. Data Preprocessing

Run the preprocessing notebook to prepare your dataset:

```python
jupyter notebook PreProcess.ipynb
```

This notebook handles data loading, normalization, and augmentation.

### 2. Model Training

Train the semantic segmentation model using either architecture:

**Standard U-Net:**
```python
jupyter notebook Train.ipynb
```

**GRU-Enhanced U-Net:**
```python
jupyter notebook Train_with_GRU.ipynb
```

Both notebooks implement custom loss functions and learning rate schedules for optimal convergence.

### 3. Inference

Apply the trained model to new images:

```python
jupyter notebook FInal.ipynb
```

For tiled prediction on large images, use:

```python
python smooth_tiled_predictions.py
```

## Notebooks

- **PreProcess.ipynb** - Data preprocessing pipeline including augmentation and normalization
- **Train.ipynb** - Main training script with custom U-Net architecture
- **Train_with_GRU.ipynb** - Alternative training with GRU layers for temporal consistency
- **FInal.ipynb** - Inference and visualization of segmentation results

## Pre-trained Models

The repository includes a pre-trained model:

- **Unetcustom_TotalLoss_SoftmaxAdam250LRschedule2000Decay1e4.hdf5** - Custom U-Net trained with total loss, Adam optimizer (LR=250), learning rate schedule (2000 steps), and weight decay (1e-4)

## Results

The trained models achieve high-quality semantic segmentation on satellite imagery, effectively classifying multiple classes including roads, buildings, vegetation, vehicles, and pedestrians. Detailed performance metrics and visualizations can be found in the inference notebooks.

## Citation

If you use this code or models in your research, please cite:

```bibtex
@software{balakavi2025usra,
  author = {Balakavi, Sai},
  title = {USRA: Urban Semantic Segmentation Research},
  year = {2025},
  url = {https://github.com/bsvskashyap/usra}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Sai Balakavi

## Author

Sai Balakavi - [bsvskashyap](https://github.com/bsvskashyap)

## Acknowledgments

This work was conducted as part of burnt area mapping using satellite imagery scene understanding research. Special thanks to the open-source community for providing foundational tools and datasets.
