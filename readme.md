# Adaptive Ternary Quantization (ATQ) for Sustainable AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **Adaptive Ternary Quantization (ATQ)** for sustainable AI systems, as described in our research paper "Sustainable Multimodal AI via Adaptive Ternary Quantization" (2025).

## Overview

ATQ is a novel quantization technique that restricts neural network weights to just three values {-1, 0, +1} while maintaining competitive accuracy. By incorporating dynamic, layer-specific thresholding, selective gradient routing, and precision-aware allocation, ATQ achieves significant memory and computational efficiency without substantial accuracy degradation.

### Key Features

- **Extreme Memory Reduction**: 16x compression ratio with bit-packing
- **High Accuracy Retention**: 85.7% accuracy on Fashion-MNIST (just 7.3% below full-precision baseline)
- **Progressive Sparsity**: Gradual sparsity targeting during training
- **Residual Precision Boosting**: Selective preservation of critical weights

## Results

Our implementation demonstrates that ATQ can achieve:

- **85.7%** test accuracy on Fashion-MNIST (vs. 93.0% for the full-precision baseline)
- **16x** theoretical memory reduction through bit-packing
- **High sparsity** in quantized layers (up to 95.67% zeros in some layers)

## Repository Structure

```
ATQ_MULTIMODAL/
├── atq/                  # Core ATQ implementation
│   ├── bit_packing.py    # Memory-efficient representation
│   ├── layers.py         # Ternary linear layers
│   ├── precision_boost.py # Residual Precision Boosting
│   ├── quantizers.py     # Adaptive ternary quantization
│   └── routing.py        # Selective gradient routing
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
└── evaluate.py           # Evaluation script
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/atq-multimodal.git
cd atq-multimodal

# Create a virtual environment (required)
python -m venv atq_multimodal
source atq_multimodal/bin/activate  # On Windows: atq_multimodal\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Make sure to create and activate the virtual environment before running any code. This ensures that all dependencies are installed in an isolated environment and don't interfere with other projects.

## Dataset

This implementation uses the Fashion-MNIST dataset, which will be automatically downloaded when running the training script. If you prefer to download it manually:

```bash
# Create data directory
mkdir -p data/FashionMNIST

# You can then place the downloaded dataset in this directory
# The datasets.py script will look for it before downloading
```

Fashion-MNIST consists of 60,000 training images and 10,000 test images of fashion items, each 28x28 pixels in grayscale. More information: [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)

## Usage

### Training

To train the ATQ model on Fashion-MNIST:

```bash
python train.py --dataset fashion_mnist --batch-size 256 --epochs 25 --use-rpb --distill --sparsity 0.3
```

Parameters:

- `--dataset`: Dataset to use ('fashion_mnist' or 'mnist')
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate (default: 0.001)
- `--use-rpb`: Enable Residual Precision Boosting
- `--distill`: Enable knowledge distillation from baseline model
- `--sparsity`: Target sparsity (0-1)

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --dataset fashion_mnist --visualize
```

Parameters:

- `--dataset`: Dataset to use ('fashion_mnist' or 'mnist')
- `--batch-size`: Batch size for evaluation
- `--visualize`: Enable weight distribution visualization

## Extending to Multimodal

This repository includes placeholder implementations for extending ATQ to multimodal data:

1. `text_encoder.py`: Framework for applying ATQ to text data
2. `fusion.py`: Cross-modal fusion with ternary weights
3. `multimodal_classifier.py`: Combined classifier for multiple modalities

Full multimodal implementation is planned for future releases.

## Visualizations

The training and evaluation scripts generate various visualizations:

- **accuracy_comparison.png**: Compares accuracy between models
- **sparsity_schedule.png**: Shows progressive sparsity during training
- **training_curve.png**: Training and validation accuracy over epochs
- **ternary_distribution.png**: Distribution of -1, 0, and +1 weights

## Citation

If you use this code in your research, please cite:

```bibtex
@article{atq2025,
  title={Sustainable Multimodal AI via Adaptive Ternary Quantization},
  author={Your Name},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Fashion-MNIST dataset by Zalando Research
- PyTorch framework
