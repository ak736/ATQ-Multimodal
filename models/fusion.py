import torch
import torch.nn as nn
import torch.nn.functional as F
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing

class MultimodalFusion(nn.Module):
    """
    Fusion module for combining features from multiple modalities using ATQ
    """
    def __init__(self, input_dims, output_dim, use_rpb=True):
        """
        Initialize multimodal fusion module
        
        Args:
            input_dims: Dictionary of {modality_name: dimension} pairs
            output_dim: Output dimension after fusion
            use_rpb: Whether to use Residual Precision Boosting
        """
        super(MultimodalFusion, self).__init__()
        self.input_dims = input_dims
        self.modalities = list(input_dims.keys())
        self.output_dim = output_dim
        self.use_rpb = use_rpb
        
        # Create projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.projections[modality] = nn.Linear(dim, output_dim)
        
        # Calculate total dimension after concatenation
        self.total_dim = output_dim * len(input_dims)
        
        # Fusion layer with ATQ
        if use_rpb:
            self.fusion_quant = nn.Sequential(
                ResidualPrecisionBoostLinear(self.total_dim, output_dim, precision_ratio=0.1),
                nn.ReLU(),
                ResidualPrecisionBoostLinear(output_dim, output_dim, precision_ratio=0.1)
            )
        else:
            self.fusion_quant = nn.Sequential(
                TernaryLinear(self.total_dim, output_dim),
                nn.ReLU(),
                TernaryLinear(output_dim, output_dim)
            )
        
        # Full-precision fusion path
        self.fusion_full = nn.Sequential(
            nn.Linear(self.total_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Learnable mixing parameter
        self.mix_ratio = nn.Parameter(torch.tensor(0.75))
        
        # Attention for weighted fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, inputs):
        """
        Forward pass for multimodal fusion
        
        Args:
            inputs: Dictionary of {modality_name: tensor} pairs
                   where each tensor has shape (batch_size, input_dim)
        
        Returns:
            Fused representation of shape (batch_size, output_dim)
        """
        # Project each modality to common dimension
        projected = {}
        for modality, tensor in inputs.items():
            if modality in self.projections:
                projected[modality] = self.projections[modality](tensor)
        
        # Concatenate all projected features
        concat_features = torch.cat([projected[m] for m in self.modalities if m in projected], dim=1)
        
        # Apply selective gradient routing
        concat_route = apply_selective_routing(concat_features, threshold=0.05)
        
        # Get outputs from both paths
        out_quant = self.fusion_quant(concat_route)
        out_full = self.fusion_full(concat_features)
        
        # Mix outputs
        mix = torch.sigmoid(self.mix_ratio)
        out = mix * out_quant + (1 - mix) * out_full
        
        return out