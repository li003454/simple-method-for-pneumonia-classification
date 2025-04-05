# Simple Method for Pneumonia Classification

This repository contains a machine learning project focused on classifying chest X-ray images to detect pneumonia. The project uses transfer learning with popular CNN architectures such as ResNet and DenseNet.

## Features

- Implementation of transfer learning using ResNet-50, ResNet-101, and DenseNet-121
- Advanced data augmentation techniques
- Handling class imbalance with weighted loss functions
- Test-time augmentation (TTA) to improve inference accuracy
- Model ensembling for improved performance
- Comprehensive evaluation metrics

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- torchvision
- scikit-learn
- matplotlib
- pandas
- kagglehub

### Installation

```bash
git clone https://github.com/li003454/simple-method-for-pneumonia-classification.git
cd simple-method-for-pneumonia-classification
pip install -r requirements.txt  # If a requirements file is available
```

### Usage

The project can be run using the Python script or Jupyter notebook:

```bash
python transfer_learning.py
```

## Dataset

The project uses the Chest X-Ray Pneumonia dataset from Kaggle by Paul Mooney, containing normal and pneumonia chest X-ray images.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 