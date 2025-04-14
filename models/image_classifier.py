import torch
import torch.nn as nn
import torch.nn.functional as F
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing

class ATQImageClassifier(nn.Module):
    """
    Enhanced image classifier using Adaptive Ternary Quantization
    with single path architecture for efficiency
    """
    def __init__(self, num_classes=10, input_channels=1, use_rpb=True, sparsity_target=0.3, hidden_size=128):
        super(ATQImageClassifier, self).__init__()
        self.use_rpb = use_rpb
        self.sparsity_target = sparsity_target
        
        # Feature extraction (unchanged - standard CNN layers)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Calculate flattened size
        flat_size = 64 * 7 * 7  # After two 2x2 pooling operations: 28/2/2 = 7
        
        # Single quantized path - no dual path, more efficient
        if use_rpb:
            self.classifier = nn.Sequential(
                ResidualPrecisionBoostLinear(flat_size, hidden_size, precision_ratio=0.05, sparsity_target=sparsity_target),
                nn.ReLU(),
                nn.Dropout(0.3),
                ResidualPrecisionBoostLinear(hidden_size, num_classes, precision_ratio=0.1, sparsity_target=sparsity_target)
            )
        else:
            self.classifier = nn.Sequential(
                TernaryLinear(flat_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                TernaryLinear(hidden_size, num_classes)
            )
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply selective gradient routing for stable training
        features_routed = apply_selective_routing(features, threshold=0.05, importance_factor=0.7)
        
        # Classify using quantized layers
        out = self.classifier(features_routed)
        
        return out
    
    # Method to extract features (useful for multimodal fusion)
    def extract_features(self, x):
        return self.features(x)