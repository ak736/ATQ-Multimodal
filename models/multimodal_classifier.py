import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_classifier import ATQImageClassifier
from .text_encoder import ATQTextEncoder
from .fusion import MultimodalFusion
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing

class ATQMultimodalClassifier(nn.Module):
    """
    Multimodal classifier using Adaptive Ternary Quantization
    """
    def __init__(self, num_classes=10, vocab_size=10000, use_rpb=True):
        super(ATQMultimodalClassifier, self).__init__()
        self.use_rpb = use_rpb
        
        # Create encoders for each modality
        self.image_encoder = ATQImageClassifier(num_classes=num_classes, use_rpb=use_rpb)
        self.text_encoder = ATQTextEncoder(vocab_size=vocab_size, use_rpb=use_rpb)
        
        # Feature dimensions from each encoder
        self.image_dim = 48 * 7 * 7  # From image encoder's feature extractor
        self.text_dim = 128  # From text encoder's output
        
        # Create fusion module
        self.fusion = MultimodalFusion(
            input_dims={'image': self.image_dim, 'text': self.text_dim},
            output_dim=256,
            use_rpb=use_rpb
        )
        
        # Classification layers
        if use_rpb:
            self.classifier_quant = nn.Sequential(
                ResidualPrecisionBoostLinear(256, 128, precision_ratio=0.1),
                nn.ReLU(),
                ResidualPrecisionBoostLinear(128, num_classes, precision_ratio=0.2)
            )
        else:
            self.classifier_quant = nn.Sequential(
                TernaryLinear(256, 128),
                nn.ReLU(),
                TernaryLinear(128, num_classes)
            )
        
        # Full-precision classification path
        self.classifier_full = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Mixing parameter
        self.mix_ratio = nn.Parameter(torch.tensor(0.75))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, image, text, text_lengths=None):
        """
        Forward pass for multimodal classifier
        
        Args:
            image: Image tensor of shape (batch_size, channels, height, width)
            text: Text tensor of shape (batch_size, sequence_length)
            text_lengths: Optional lengths of text sequences for packed padded sequence
        
        Returns:
            Classification logits
        """
        # Extract features from each modality
        image_features = self.image_encoder.extract_features(image)
        text_features = self.text_encoder(text, text_lengths)
        
        # Prepare inputs for fusion
        multimodal_inputs = {
            'image': image_features,
            'text': text_features
        }
        
        # Fuse modalities
        fused_features = self.fusion(multimodal_inputs)
        fused_features = self.dropout(fused_features)
        
        # Apply selective gradient routing
        fused_route = apply_selective_routing(fused_features, threshold=0.05)
        
        # Classification with both paths
        out_quant = self.classifier_quant(fused_route)
        out_full = self.classifier_full(fused_features)
        
        # Mix outputs
        mix = torch.sigmoid(self.mix_ratio)
        out = mix * out_quant + (1 - mix) * out_full
        
        return out