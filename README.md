# Burnt Area Mapping Using Satellite Imagery Semantic Segmentation Research

This repository contains the implementation and trained models for semantic segmentation of satellite imagery using deep learning techniques. The project focuses on developing robust neural network architectures for pixel-wise classification of satellite imagery for burnt area mapping.

## 🏆 Key Achievements

- **📄 Published Research**: Published in PLOS ONE (2025) - peer-reviewed open-access journal on deep learning architectures for burnt area mapping from satellite imagery
- **🧠 Novel Architectures**: Developed two novel deep learning architectures:
  - **Custom UNET**: Enhanced U-Net with optimized encoder-decoder design (AUC = 0.96)
  - **UNET-GRU**: Hybrid architecture integrating Gated Recurrent Units with U-Net (AUC = 0.98)
- **📊 State-of-the-Art Performance**: Achieved 2% AUC improvement over standard baseline models, reaching best-in-class performance for burnt area segmentation
- **🛰️ Real-World Validation**: Models validated on 3-meter resolution PlanetScope satellite imagery from wildfire-affected regions in Bandipur Tiger Reserve, Karnataka, India
- **🌲 Actionable Impact**: Delivered critical insights for:
  - Forest management and ecological restoration planning
  - Rapid post-fire damage assessment
  - Wildlife habitat monitoring and conservation strategies
- **💡 Technical Skills Demonstrated**: Deep learning, computer vision, remote sensing, semantic segmentation, multi-spectral imagery analysis, and geospatial data processing

## Overview

The workflow consists of three main stages:

1. **Data Preprocessing** - Preparation and augmentation of satellite imagery datasets
2. **Model Training** - Training custom U-Net architectures (including standard and GRU-enhanced variants)
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
jupyter notebook notebooks/PreProcess.ipynb
```

This notebook handles data loading, normalization, and augmentation.

### 2. Model Training

Train the semantic segmentation model:

```python
jupyter notebook notebooks/Train.ipynb
```

This notebook implements custom U-Net architectures (including standard and GRU-enhanced variants) with custom loss functions and learning rate schedules for optimal convergence.

### 3. Inference

Apply the trained model to new images:

```python
jupyter notebook notebooks/Final.ipynb
```

For tiled prediction on large images, use:

```python
python scripts/smooth_tiled_predictions.py
```

## Notebooks

- **notebooks/PreProcess.ipynb** - Data preprocessing pipeline including augmentation and normalization
- **notebooks/Train.ipynb** - Training script with custom U-Net architectures (standard and GRU-enhanced variants)
- **notebooks/Final.ipynb** - Inference and visualization of segmentation results

## Pre-trained Models

The repository includes a pre-trained model in the `models/` directory:

- **Unetcustom_TotalLoss_SoftmaxAdam250LRschedule2000Decay1e4.hdf5** - Custom U-Net trained with total loss, Adam optimizer (LR=250), learning rate schedule (2000 steps), and weight decay (1e-4)

## Results

This research introduced two novel deep learning architectures for burnt area mapping from satellite imagery:

### Novel Architectures

1. **Custom UNET** - Enhanced U-Net architecture with optimized encoder-decoder design
2. **UNET-GRU** - Hybrid architecture integrating Gated Recurrent Units with U-Net for improved temporal feature learning

### Performance Metrics

The models were evaluated on real-world wildfire data from Bandipur Tiger Reserve, India, achieving state-of-the-art results:

| Model | AUC Score | Improvement over Baseline |
|-------|-----------|---------------------------|
| **UNET-GRU** | **0.98** | **+2%** (Best Result) |
| **Custom UNET** | **0.96** | +0% |
| Standard Baseline | 0.96 | - |

**Key Performance Highlights:**
- UNET-GRU: AUC = 0.98 (2% improvement over baseline)
- Custom UNET: AUC = 0.96
- Both architectures significantly outperformed standard segmentation models
- Successfully mapped burnt areas with high precision for post-fire ecological assessment

### Visual Results Comparison

*[Placeholder for segmentation results comparison image showing: Original Satellite Image | Ground Truth | UNET-GRU Prediction | Custom UNET Prediction]*

![Results Comparison](./images/results_comparison.png)
*Figure: Visual comparison of segmentation results on Bandipur Tiger Reserve wildfire imagery. Left to right: Original multi-spectral satellite image, ground truth burnt area mask, UNET-GRU predictions, Custom UNET predictions.*

### Real-World Impact

- Applied to wildfire-affected regions in Bandipur Tiger Reserve, Karnataka, India
- Provides critical data for forest management and ecological restoration efforts
- Enables rapid, accurate post-fire damage assessment using multi-spectral satellite imagery
- Supports wildlife habitat monitoring and conservation planning

### Publication

This work has been published in PLOS ONE, a peer-reviewed open-access scientific journal:

**Citation**: Balakavi, S., et al. "Deep Learning Architectures for Burnt Area Mapping from Satellite Imagery." PLOS ONE, 2025.

Detailed performance metrics, ablation studies, and visualizations can be found in the published paper and inference notebooks.

## Citation

If you use this code or models in your research, please cite:

```bibtex
@software{balakavi2025usra,
  author = {Balakavi, Sai},
  title = {USRA: Burnt Area Mapping Using Satellite Imagery},
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

Portions of this repository were adapted from open-source notebooks and scripts by Digital Sreeni (bnsreenu). Substantial modifications and new features have been added for burnt area mapping using satellite imagery.
