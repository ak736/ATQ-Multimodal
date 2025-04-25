import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing
from atq.quantizers import adaptive_ternary_quantization

class MixedPrecisionATQ:
    """
    Enhanced Adaptive Ternary Quantization with mixed precision allocation
    and gradual quantization schedules.
    """
    
    @staticmethod
    def get_layer_importance(model, layer_name, default_importance=1.0):
        """
        Determine layer importance based on heuristics
        
        Args:
            model: The model containing the layer
            layer_name: Name of the layer
            default_importance: Default importance value
            
        Returns:
            Importance score (higher = needs more precision)
        """
        # Layers that critically need precision
        critical_keywords = ['fusion', 'cross_attention', 'projector', 'final']
        # Layers that need medium precision
        medium_keywords = ['attention', 'embed', 'pool']
        # Layers that can handle more sparsity
        low_keywords = ['intermediate', 'ffn', 'conv']
        
        # Check if layer is critical
        if any(keyword in layer_name for keyword in critical_keywords):
            return 2.0
        # Check if layer needs medium precision
        elif any(keyword in layer_name for keyword in medium_keywords):
            return 1.5
        # Check if layer can be more sparse
        elif any(keyword in layer_name for keyword in low_keywords):
            return 0.8
        
        return default_importance
    
    @staticmethod
    def get_precision_ratio(importance, base_ratio=0.05, max_ratio=0.25):
        """
        Calculate precision ratio based on layer importance
        
        Args:
            importance: Layer importance score
            base_ratio: Base precision ratio
            max_ratio: Maximum precision ratio
            
        Returns:
            Precision ratio for ResidualPrecisionBoostLinear
        """
        ratio = base_ratio * importance
        return min(max_ratio, ratio)
    
    @staticmethod
    def get_sparsity_target(importance, base_sparsity=0.3, min_sparsity=0.1):
        """
        Calculate sparsity target based on layer importance
        
        Args:
            importance: Layer importance score
            base_sparsity: Base sparsity target
            min_sparsity: Minimum sparsity target
            
        Returns:
            Sparsity target for adaptive ternary quantization
        """
        # Inverse relationship - more important layers get less sparsity
        sparsity = base_sparsity / importance
        return max(min_sparsity, sparsity)
    
    @classmethod
    def calculate_quantization_params(cls, model, layer_name, epoch, total_epochs, 
                                     target_sparsity, initial_ratio=0.05):
        """
        Calculate quantization parameters based on training progress
        
        Args:
            model: The model containing the layer
            layer_name: Name of the layer
            epoch: Current epoch
            total_epochs: Total training epochs
            target_sparsity: Target sparsity at the end of training
            initial_ratio: Initial precision ratio
            
        Returns:
            precision_ratio, current_sparsity
        """
        # Determine layer importance
        importance = cls.get_layer_importance(model, layer_name)
        
        # Calculate base precision ratio based on importance
        precision_ratio = cls.get_precision_ratio(importance, base_ratio=initial_ratio)
        
        # Calculate target sparsity based on importance
        final_sparsity = cls.get_sparsity_target(importance, base_sparsity=target_sparsity)
        
        # Calculate gradual quantization schedule - start with lower sparsity
        progress = min(1.0, epoch / (total_epochs * 0.8))
        initial_sparsity = min(0.1, final_sparsity)
        current_sparsity = initial_sparsity + progress * (final_sparsity - initial_sparsity)
        
        return precision_ratio, current_sparsity
    
    @staticmethod
    def update_model_quantization(model, epoch, total_epochs, vision_threshold=0.3, text_threshold=0.2):
        """
        Update quantization parameters across the model based on training progress
        
        Args:
            model: The model to update
            epoch: Current epoch
            total_epochs: Total training epochs
            vision_threshold: Target sparsity for vision components
            text_threshold: Target sparsity for text components
        """
        # Set model's epoch trackers if available
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch, total_epochs)
        
        # Iterate through named modules to update quantization params
        for name, module in model.named_modules():
            # Update ResidualPrecisionBoostLinear layers
            if isinstance(module, ResidualPrecisionBoostLinear):
                # Determine if this is a vision or text component
                component_type = 'vision' if 'image' in name else 'text'
                threshold = vision_threshold if component_type == 'vision' else text_threshold
                
                # Calculate quantization parameters
                precision_ratio, current_sparsity = MixedPrecisionATQ.calculate_quantization_params(
                    model, name, epoch, total_epochs, threshold
                )
                
                # Update module parameters
                module.precision_ratio = precision_ratio
                module.sparsity_target = current_sparsity


class GradualQuantizationScheduler:
    """
    Scheduler to gradually increase quantization during training
    """
    def __init__(self, model, total_epochs, vision_sparsity=0.3, text_sparsity=0.2,
                 warmup_epochs=5, final_epochs=None, verbose=False):
        """
        Initialize scheduler
        
        Args:
            model: The model to apply gradual quantization to
            total_epochs: Total number of training epochs
            vision_sparsity: Target sparsity for vision components
            text_sparsity: Target sparsity for text components
            warmup_epochs: Number of epochs with minimal quantization
            final_epochs: Number of epochs with final quantization (None = auto)
            verbose: Whether to print quantization updates
        """
        self.model = model
        self.total_epochs = total_epochs
        self.vision_sparsity = vision_sparsity
        self.text_sparsity = text_sparsity
        self.warmup_epochs = warmup_epochs
        self.final_epochs = final_epochs or max(2, int(total_epochs * 0.2))
        self.verbose = verbose
        
        # Initial sparsity values are very low during warmup
        self.initial_vision_sparsity = 0.05
        self.initial_text_sparsity = 0.05
        
        # Track the effective sparsity at each epoch
        self.vision_sparsity_schedule = self._create_schedule(
            self.initial_vision_sparsity, self.vision_sparsity
        )
        self.text_sparsity_schedule = self._create_schedule(
            self.initial_text_sparsity, self.text_sparsity
        )
    
    def _create_schedule(self, initial_value, final_value):
        """Create sparsity schedule over epochs"""
        schedule = []
        
        # Warmup phase - minimal quantization
        for _ in range(self.warmup_epochs):
            schedule.append(initial_value)
        
        # Gradual phase - linear increase in quantization
        gradual_epochs = self.total_epochs - self.warmup_epochs - self.final_epochs
        for i in range(gradual_epochs):
            progress = (i + 1) / gradual_epochs
            value = initial_value + progress * (final_value - initial_value)
            schedule.append(value)
        
        # Final phase - maintain target quantization
        for _ in range(self.final_epochs):
            schedule.append(final_value)
        
        return schedule
    
    def step(self, epoch):
        """
        Update model quantization based on current epoch
        
        Args:
            epoch: Current epoch (0-indexed)
        """
        if epoch >= len(self.vision_sparsity_schedule):
            # Use final values if beyond schedule length
            current_vision_sparsity = self.vision_sparsity
            current_text_sparsity = self.text_sparsity
        else:
            # Get scheduled values
            current_vision_sparsity = self.vision_sparsity_schedule[epoch]
            current_text_sparsity = self.text_sparsity_schedule[epoch]
        
        # Update model's quantization parameters
        MixedPrecisionATQ.update_model_quantization(
            self.model, epoch, self.total_epochs,
            vision_threshold=current_vision_sparsity,
            text_threshold=current_text_sparsity
        )
        
        # Print information if verbose
        if self.verbose:
            print(f"Epoch {epoch+1}: Vision sparsity = {current_vision_sparsity:.3f}, "
                  f"Text sparsity = {current_text_sparsity:.3f}")
        
        return current_vision_sparsity, current_text_sparsity


class PrecisionControlledLinear(nn.Module):
    """
    Linear layer with controlled precision based on importance
    """
    def __init__(self, in_features, out_features, importance=1.0, 
                 base_sparsity=0.3, base_precision_ratio=0.05,
                 bias=True, use_rpb=True):
        """
        Initialize precision-controlled linear layer
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            importance: Importance factor for precision allocation
            base_sparsity: Base sparsity target
            base_precision_ratio: Base precision ratio
            bias: Whether to include bias
            use_rpb: Whether to use ResidualPrecisionBoost
        """
        super(PrecisionControlledLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.importance = importance
        self.use_rpb = use_rpb
        
        # Calculate precision parameters based on importance
        precision_ratio = MixedPrecisionATQ.get_precision_ratio(
            importance, base_ratio=base_precision_ratio
        )
        sparsity_target = MixedPrecisionATQ.get_sparsity_target(
            importance, base_sparsity=base_sparsity
        )
        
        # Create appropriate linear layer
        if use_rpb:
            self.linear = ResidualPrecisionBoostLinear(
                in_features, out_features, 
                precision_ratio=precision_ratio,
                sparsity_target=sparsity_target,
                bias=bias
            )
        else:
            self.linear = TernaryLinear(
                in_features, out_features, bias=bias
            )
    
    def forward(self, x):
        return self.linear(x)


# Example usage in a model:
class EnhancedATQTransformerLayer(nn.Module):
    """
    Transformer layer using enhanced ATQ with mixed precision allocation
    """
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, 
                 use_rpb=True, base_sparsity=0.3, layer_idx=0, total_layers=4):
        super(EnhancedATQTransformerLayer, self).__init__()
        
        # Calculate layer importance - later layers and attention need more precision
        self.layer_idx = layer_idx
        layer_progress = layer_idx / max(1, total_layers - 1)  # 0 to 1
        layer_importance = 1.0 + layer_progress  # 1.0 to 2.0
        
        # Higher importance for attention compared to feedforward
        attn_importance = layer_importance * 1.2
        ff_importance = layer_importance * 0.8
        
        # Create precision-controlled attention components
        self.query = PrecisionControlledLinear(
            embed_dim, embed_dim, importance=attn_importance,
            base_sparsity=base_sparsity, use_rpb=use_rpb
        )
        self.key = PrecisionControlledLinear(
            embed_dim, embed_dim, importance=attn_importance,
            base_sparsity=base_sparsity, use_rpb=use_rpb
        )
        self.value = PrecisionControlledLinear(
            embed_dim, embed_dim, importance=attn_importance,
            base_sparsity=base_sparsity, use_rpb=use_rpb
        )
        self.attn_out = PrecisionControlledLinear(
            embed_dim, embed_dim, importance=attn_importance * 1.1,  # Even more important
            base_sparsity=base_sparsity, use_rpb=use_rpb
        )
        
        # Create precision-controlled feedforward components
        self.ff1 = PrecisionControlledLinear(
            embed_dim, dim_feedforward, importance=ff_importance,
            base_sparsity=base_sparsity, use_rpb=use_rpb
        )
        self.ff2 = PrecisionControlledLinear(
            dim_feedforward, embed_dim, importance=ff_importance * 1.2,  # Output is more important
            base_sparsity=base_sparsity, use_rpb=use_rpb
        )
        
        # Layer normalization and dropout (full precision)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-head attention parameters
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    
    def _attention(self, q, k, v, mask=None):
        # Multi-head attention calculation
        batch_size = q.size(0)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return out
    
    def forward(self, x, mask=None):
        # Self-attention with mixed precision
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Apply selective gradient routing with adaptive threshold
        threshold = 0.05 * (1.0 - self.layer_idx / 10)  # Lower threshold for later layers
        threshold = max(0.01, threshold)  # Minimum threshold
        
        q = apply_selective_routing(q, threshold=threshold)
        k = apply_selective_routing(k, threshold=threshold)
        v = apply_selective_routing(v, threshold=threshold)
        
        # Compute self-attention
        attn_output = self._attention(q, k, v, mask)
        attn_output = self.attn_out(attn_output)
        
        # First residual connection
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network with mixed precision
        ff_output = self.ff1(x)
        ff_output = F.gelu(ff_output)  # GELU instead of ReLU
        ff_output = self.dropout(ff_output)
        ff_output = self.ff2(ff_output)
        
        # Second residual connection
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x