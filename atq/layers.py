import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizers import adaptive_ternary_quantization

class TernaryLinear(nn.Module):
    """
    Linear layer with Adaptive Ternary Quantization - simplified for consistency
    """
    def __init__(self, in_features, out_features, bias=True):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Regular weights and scaling parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.alpha = nn.Parameter(torch.Tensor(1))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.alpha, 1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        # Quantize weights to ternary values - same behavior in train and eval
        w_ternary, alpha = adaptive_ternary_quantization(
            self.weight, 
            alpha=self.alpha,
        )
        
        # Use quantized weights for forward pass
        return F.linear(input, w_ternary * alpha, self.bias)