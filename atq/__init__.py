# Import and expose key components
from .quantizers import adaptive_ternary_quantization
from .layers import TernaryLinear
from .routing import apply_selective_routing, SelectiveGradientRouting
from .precision_boost import ResidualPrecisionBoostLinear

__all__ = [
    'adaptive_ternary_quantization',
    'TernaryLinear',
    'SelectiveGradientRouting',
    'apply_selective_routing',
    'ResidualPrecisionBoostLinear'
]