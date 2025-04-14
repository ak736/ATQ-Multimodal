# precision_boost.py - Enhanced implementation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizers import adaptive_ternary_quantization

class ResidualPrecisionBoostLinear(nn.Module):
    """
    Enhanced Residual Precision Boosting Linear layer
    Uses a small portion of weights in full precision for improved accuracy
    """
    def __init__(self, in_features, out_features, precision_ratio=0.05, bias=True, sparsity_target=0.3):
        super(ResidualPrecisionBoostLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision_ratio = precision_ratio
        self.sparsity_target = sparsity_target
        
        # Main quantized weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.alpha = nn.Parameter(torch.Tensor(1))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Create fixed precision mask rather than learnable
        # This simplifies the implementation and avoids extra parameters
        self.register_buffer('precision_mask', torch.zeros(out_features, in_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize main weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.alpha, 1.0)
        
        # Initialize bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize precision mask
        with torch.no_grad():
            # Get a flattened view of weight magnitudes
            abs_weights = torch.abs(self.weight)
            flat_abs_weights = abs_weights.view(-1)
            
            # Find top k% indices by magnitude (most important weights)
            k = int(self.precision_ratio * flat_abs_weights.numel())
            _, indices = torch.topk(flat_abs_weights, k)
            
            # Set precision mask for these indices to 1.0
            flat_mask = self.precision_mask.view(-1)
            flat_mask[indices] = 1.0
    
    def forward(self, input):
        # Quantize weights to ternary values
        w_ternary, alpha = adaptive_ternary_quantization(
            self.weight, 
            alpha=self.alpha,
            sparsity_target=self.sparsity_target
        )
        
        # Full precision contribution for critical weights
        # Element-wise multiply with precision mask
        w_mixed = w_ternary * alpha * (1 - self.precision_mask) + self.weight * self.precision_mask
        
        return F.linear(input, w_mixed, self.bias)
    
    def get_quantized_weights(self):
        """
        Get the ternary quantized weights and scaling factor
        
        This method is used for analysis and bit-packing
        
        Returns:
            Tuple of (ternary_weights, scaling_factor)
        """
        # Quantize weights to ternary values
        w_ternary, alpha = adaptive_ternary_quantization(
            self.weight, 
            alpha=self.alpha,
            sparsity_target=self.sparsity_target
        )
        
        return w_ternary, alpha