# Sustainable Scaling for Multimodal AI through Adaptive Ternary Quantization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **Adaptive Ternary Quantization (ATQ)** for sustainable AI systems, including multimodal learning.

## Overview

ATQ is a novel quantization technique that restricts neural network weights to just three values {-1, 0, +1} while maintaining competitive accuracy. By incorporating dynamic, layer-specific thresholding, selective gradient routing, and precision-aware allocation, ATQ achieves significant memory and computational efficiency without substantial accuracy degradation.


### Key Features

- **Extreme Memory Reduction**: 8-10x compression ratio compared to full-precision models
- **High Accuracy Retention**: 85.7% accuracy on Fashion-MNIST (just 7.3% below full-precision baseline)
- **Progressive Sparsity**: Gradual sparsity targeting during training
- **Residual Precision Boosting**: Selective preservation of critical weights
- **Multimodal Support**: Extended framework for image-text multimodal tasks
- **Mixed-Precision Allocation**: Layer-specific precision based on importance

## Results

Our implementation demonstrates that ATQ can achieve:

- **85.7%** test accuracy on Fashion-MNIST (vs. 93.0% for the full-precision baseline)
- **16x** theoretical memory reduction through bit-packing
- **High sparsity** in quantized layers (up to 95.67% zeros in some layers)

For multimodal tasks on Flickr8k dataset:
- **Model size reduction**: 15-22MB (vs. 75-100MB for full-precision models)
- **Inference time**: ~195-277ms per sample
- With further optimization, potential for **15-25% Recall@1** and **40-50% Recall@5**

## Repository Structure

```
ATQ_MULTIMODAL/
├── atq/                  # Core ATQ implementation
│   ├── bit_packing.py    # Memory-efficient representation
│   ├── layers.py         # Ternary linear layers
│   ├── precision_boost.py # Residual Precision Boosting
│   ├── quantizers.py     # Adaptive ternary quantization
│   └── routing.py        # Selective gradient routing
│   └── mixed_precision_atq.py    # Enhanced mixed-precision ATQ
├── data/                 # Data handling utilities
│   ├── datasets.py       # Dataset loading
│   └── multimodal_data.py # Multimodal dataset handling
├── models/               # Model architectures
│   ├── fusion.py         # Multimodal fusion module
│   ├── image_classifier.py # Image classifier
│   ├── multimodal_classifier.py # Multimodal classifier
│   └── text_encoder.py   # Text encoder
├── utils/                # Utility functions
│   ├── metrics.py        # Performance measurement
│   └── visualization.py  # Visualization utilities
├── plots/                # Generated visualizations
├── checkpoints/          # Saved model weights
├── train.py              # Training script
├── train_multimodal.py   # Training script for multimodal
└── evaluate.py           # Evaluation script
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- nltk (for text processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/atq-multimodal.git
cd atq-multimodal

# Create a virtual environment (required)
python3 -m venv atq_multimodal
source atq_multimodal/bin/activate  # On Windows: atq_multimodal\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt
```

Make sure to create and activate the virtual environment before running any code. This ensures that all dependencies are installed in an isolated environment and don't interfere with other projects.

## Dataset

This implementation uses the Fashion-MNIST dataset, which will be automatically downloaded when running the training script. If you prefer to download it manually:

### Fashion-MNIST
This implementation uses the Fashion-MNIST dataset for single-modality experiments, which will be automatically downloaded when running the training script. 

### Flickr8k

For multimodal experiments, we use the Flickr8k dataset, which contains 8,000 images with 5 captions each. The dataset will be automatically downloaded when running the multimodal training script.

## Usage

### Training

### Training Single-Modality Model
To train the ATQ model on Fashion-MNIST:

```bash
python3 train.py --dataset fashion_mnist --batch-size 256 --epochs 25 --use-rpb --distill --sparsity 0.3
```

Parameters:

- `--dataset`: Dataset to use ('fashion_mnist' or 'mnist')
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate (default: 0.001)
- `--use-rpb`: Enable Residual Precision Boosting
- `--distill`: Enable knowledge distillation from baseline model
- `--sparsity`: Target sparsity (0-1)

### Training Multimodal Model
To train the ATQ multimodal model on Flickr8k:

```bash
python3 train_multimodal.py --device mps --batch-size 16 --embed-dim 192 --hidden-dim 384 --epochs 10 --learning-rate 5e-5 --image-size 160 --use-residual --reinit-model --gradual-quant --warmup-epochs 2 --contrastive-reg 0.05
```

Parameters:

- `--device`: Device to use ('cpu', 'cuda', 'mps')
- `--batch-size`: Batch size for training
- `--embed-dim`: Embedding dimension for joint space
- `--hidden-dim`: Hidden dimension for encoders
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate
- `--image-size`: Image size for resizing
- `--use-residual`: Enable Residual Precision Boosting
- `--reinit-model`: Reinitialize model weights
- `--gradual-quant`: Use gradual quantization schedule
- `--warmup-epochs`: Number of warmup epochs
- `--contrastive-reg`: Regularization for contrastive loss


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Fashion-MNIST dataset by Zalando Research
- Flickr8k dataset
- PyTorch framework
