# quantizers.py - Apply this enhanced implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

def adaptive_ternary_quantization(weights, alpha=None, threshold_factor=0.05, sparsity_target=0.3):
    """
    Enhanced Adaptive Ternary Quantization with direct sparsity targeting
    
    Args:
        weights: Tensor of floating-point weights
        alpha: Scaling factor (if None, computed adaptively)
        threshold_factor: Base factor for threshold (used if sparsity_target is None)
        sparsity_target: Target percentage of zeros (0-1)
        
    Returns:
        Quantized weights (ternary) and scaling factor alpha
    """
    # Compute absolute values
    abs_weights = torch.abs(weights)
    
    # Get the flattened weight magnitudes and sort them
    flat_abs_weights = abs_weights.view(-1)
    sorted_weights, _ = torch.sort(flat_abs_weights)
    
    # Find the threshold index that would give target sparsity
    threshold_idx = int(sparsity_target * sorted_weights.numel())
    
    # Use that threshold (handle edge cases)
    if threshold_idx < sorted_weights.numel() and threshold_idx > 0:
        threshold = sorted_weights[threshold_idx]
    elif threshold_idx >= sorted_weights.numel():
        # All zeros case (unlikely in practice)
        threshold = sorted_weights[-1] + 1.0
    else:
        # Fallback to classic method if index computation fails
        threshold = threshold_factor * torch.mean(abs_weights)
    
    # Create ternary weights
    w_ternary = torch.zeros_like(weights)
    w_ternary[weights > threshold] = 1.0
    w_ternary[weights < -threshold] = -1.0
    
    # Count non-zero elements for optimal scaling
    nonzero_count = torch.sum(w_ternary != 0).float()
    
    # Compute optimal scaling factor to minimize quantization error
    if nonzero_count > 0:
        # Get sum of weights with matching signs
        # This minimizes the L2 error between W and alpha*W_ternary
        optimal_alpha = torch.sum(weights * w_ternary) / nonzero_count
    else:
        # Fallback if all weights are zeros
        optimal_alpha = torch.mean(abs_weights)
    
    # Use provided alpha or computed alpha
    if alpha is None:
        alpha = optimal_alpha
    
    return w_ternary, alpha