import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing
from atq.quantizers import adaptive_ternary_quantization

class TernaryCrossAttention(nn.Module):
    """
    Cross-attention mechanism with Adaptive Ternary Quantization
    for fusing features from different modalities
    """
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads=4, 
                 dropout=0.1, use_rpb=True, sparsity_target=0.3):
        super(TernaryCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rpb = use_rpb
        
        # NEW: Gradual sparsity approach
        self.initial_sparsity = min(0.1, sparsity_target)
        self.target_sparsity = sparsity_target
        self.current_sparsity = self.initial_sparsity
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # NEW: Increased precision ratio for critical cross-modal attention
        precision_ratio_qkv = 0.15  # Higher precision for cross-modal attention
        precision_ratio_out = 0.2   # Even higher for output projection
        
        # Projection layers with ATQ
        if use_rpb:
            self.q_proj = ResidualPrecisionBoostLinear(
                query_dim, hidden_dim, precision_ratio=precision_ratio_qkv, 
                sparsity_target=self.initial_sparsity
            )
            self.k_proj = ResidualPrecisionBoostLinear(
                key_dim, hidden_dim, precision_ratio=precision_ratio_qkv, 
                sparsity_target=self.initial_sparsity
            )
            self.v_proj = ResidualPrecisionBoostLinear(
                value_dim, hidden_dim, precision_ratio=precision_ratio_qkv, 
                sparsity_target=self.initial_sparsity
            )
            self.out_proj = ResidualPrecisionBoostLinear(
                hidden_dim, hidden_dim, precision_ratio=precision_ratio_out, 
                sparsity_target=self.initial_sparsity
            )
        else:
            self.q_proj = TernaryLinear(query_dim, hidden_dim)
            self.k_proj = TernaryLinear(key_dim, hidden_dim)
            self.v_proj = TernaryLinear(value_dim, hidden_dim)
            self.out_proj = TernaryLinear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # NEW: Layer normalization for stability
        self.layer_norm_q = nn.LayerNorm(query_dim)
        self.layer_norm_k = nn.LayerNorm(key_dim)
        self.layer_norm_v = nn.LayerNorm(value_dim)
        self.layer_norm_out = nn.LayerNorm(hidden_dim)
        
        # NEW: Gating parameter to control information flow
        self.gate = nn.Parameter(torch.ones(1) * 0.8)
        
        # NEW: Customizable attention scale
        self.attention_scale = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(self.head_dim)))
    
    def update_sparsity(self, progress_ratio):
        """
        NEW: Update sparsity based on training progress
        """
        self.current_sparsity = self.initial_sparsity + progress_ratio * (self.target_sparsity - self.initial_sparsity)
        
        # Update sparsity for all projection layers
        if hasattr(self.q_proj, 'sparsity_target'):
            self.q_proj.sparsity_target = self.current_sparsity
            self.k_proj.sparsity_target = self.current_sparsity
            self.v_proj.sparsity_target = self.current_sparsity
            self.out_proj.sparsity_target = self.current_sparsity
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # NEW: Apply layer normalization before quantized operations
        query = self.layer_norm_q(query)
        key = self.layer_norm_k(key)
        value = self.layer_norm_v(value)
        
        # Linear projections with ATQ
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Apply selective gradient routing for stable training
        # NEW: Lower threshold for better gradient flow
        q = apply_selective_routing(q, threshold=0.01)
        k = apply_selective_routing(k, threshold=0.01)
        v = apply_selective_routing(v, threshold=0.01)
        
        # Reshape for multi-head attention
        # Add sequence dimension if input is 2D (batch_size x features)
        if q.dim() == 2:
            q = q.unsqueeze(1)
        if k.dim() == 2:
            k = k.unsqueeze(1)
        if v.dim() == 2:
            v = v.unsqueeze(1)
            
        # Now reshape to multi-head format
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with learnable scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        
        # Squeeze sequence dimension if it's 1
        if attn_output.size(1) == 1:
            attn_output = attn_output.squeeze(1)
            
        # Output projection
        output = self.out_proj(attn_output)
        
        # NEW: Additional layer norm after projection
        output = self.layer_norm_out(output)
        
        # NEW: Add gated residual connection if dimensions match
        if query.dim() == output.dim() and query.size(-1) == output.size(-1):
            gate_value = torch.sigmoid(self.gate)
            output = gate_value * output + (1 - gate_value) * query
            
        return output


class ModalitySpecificQuantization(nn.Module):
    """
    Implements modality-specific sparsity targeting for ATQ
    """
    def __init__(self, input_dim, output_dim, modality_name, use_rpb=True):
        super(ModalitySpecificQuantization, self).__init__()
        self.modality_name = modality_name
        self.use_rpb = use_rpb
        
        # Set modality-specific sparsity target
        if modality_name == 'image':
            # Images can tolerate higher sparsity
            self.target_sparsity = 0.3
        elif modality_name == 'text':
            # Text needs lower sparsity for semantic preservation
            self.target_sparsity = 0.2
        elif modality_name == 'fusion':
            # Fusion layers need lowest sparsity for information preservation
            self.target_sparsity = 0.15
        else:
            # Default
            self.target_sparsity = 0.25
        
        # NEW: Start with lower sparsity for better training
        self.initial_sparsity = min(0.1, self.target_sparsity)
        self.current_sparsity = self.initial_sparsity
        
        # NEW: Increased precision ratio for better feature representation
        precision_ratio = 0.2 if modality_name == 'fusion' else 0.15
        
        # Create ternary projection layer
        if use_rpb:
            self.projection = ResidualPrecisionBoostLinear(
                input_dim, output_dim, 
                precision_ratio=precision_ratio, 
                sparsity_target=self.initial_sparsity
            )
        else:
            self.projection = TernaryLinear(input_dim, output_dim)
            
        # NEW: Add normalization for better stability
        self.norm = nn.LayerNorm(output_dim)
        
        # NEW: Add activation function for non-linearity
        self.activation = nn.GELU()  # GELU instead of ReLU for better gradient flow
    
    def update_sparsity(self, progress_ratio):
        """
        NEW: Update sparsity based on training progress
        """
        self.current_sparsity = self.initial_sparsity + progress_ratio * (self.target_sparsity - self.initial_sparsity)
        
        # Update sparsity for projection layer
        if hasattr(self.projection, 'sparsity_target'):
            self.projection.sparsity_target = self.current_sparsity
    
    def forward(self, x):
        # Project features
        x = self.projection(x)
        
        # Apply normalization and activation
        x = self.norm(x)
        x = self.activation(x)
        
        return x


class MultimodalFusion(nn.Module):
    """
    Fusion module for combining features from multiple modalities using ATQ
    """
    def __init__(self, input_dims, output_dim, fusion_method='cross_attention', 
                 num_heads=4, dropout=0.1, use_rpb=True):
        super(MultimodalFusion, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        self.use_rpb = use_rpb
        
        # NEW: Start with lower sparsity for better training
        self.fusion_sparsity = 0.15  # Lower sparsity for fusion
        self.initial_sparsity = 0.05  # Start even lower
        self.current_sparsity = self.initial_sparsity
        
        # Project each modality to common dimension with modality-specific sparsity
        self.modality_projections = nn.ModuleDict({
            name: ModalitySpecificQuantization(
                dim, output_dim, name, use_rpb
            ) for name, dim in input_dims.items()
        })
        
        # Print input dimensions for debugging
        print(f"Fusion input dimensions: {input_dims}")
        print(f"Output dimension: {output_dim}")
        
        # Fusion methods
        if fusion_method == 'cross_attention':
            # Assuming two modalities (text and image) for cross-attention
            if 'text' in input_dims and 'image' in input_dims:
                # Text-to-image attention
                self.text2image = TernaryCrossAttention(
                    output_dim, output_dim, output_dim, output_dim, 
                    num_heads, dropout, use_rpb, sparsity_target=self.initial_sparsity
                )
                
                # Image-to-text attention
                self.image2text = TernaryCrossAttention(
                    output_dim, output_dim, output_dim, output_dim,
                    num_heads, dropout, use_rpb, sparsity_target=self.initial_sparsity
                )
                
                # NEW: Additional cross-modal projection for better alignment
                if use_rpb:
                    self.cross_modal_align = nn.ModuleDict({
                        'text': ResidualPrecisionBoostLinear(
                            output_dim, output_dim, 
                            precision_ratio=0.2, 
                            sparsity_target=self.initial_sparsity
                        ),
                        'image': ResidualPrecisionBoostLinear(
                            output_dim, output_dim, 
                            precision_ratio=0.2, 
                            sparsity_target=self.initial_sparsity
                        )
                    })
                
                # Final fusion layer (after attention)
                if use_rpb:
                    self.final_fusion = ResidualPrecisionBoostLinear(
                        output_dim * 2, output_dim, 
                        precision_ratio=0.2, 
                        sparsity_target=self.initial_sparsity
                    )
                else:
                    self.final_fusion = TernaryLinear(output_dim * 2, output_dim)
        
        elif fusion_method == 'concat':
            # Simple concatenation followed by projection
            if use_rpb:
                self.fusion_layer = ResidualPrecisionBoostLinear(
                    output_dim * len(input_dims), output_dim,
                    precision_ratio=0.2,
                    sparsity_target=self.initial_sparsity
                )
            else:
                self.fusion_layer = TernaryLinear(output_dim * len(input_dims), output_dim)
        
        else:  # Default to element-wise methods
            # For element-wise methods, we need to project to same dimension first
            if use_rpb:
                self.fusion_gate = ResidualPrecisionBoostLinear(
                    output_dim * len(input_dims), output_dim,
                    precision_ratio=0.2,
                    sparsity_target=self.initial_sparsity
                )
            else:
                self.fusion_gate = TernaryLinear(output_dim * len(input_dims), output_dim)
        
        # Layer normalization kept in full precision
        self.norm = nn.LayerNorm(output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # NEW: Add scaling factors for modality balancing
        self.modality_scales = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1) * 1.0) 
            for name in input_dims.keys()
        })
    
    def update_sparsity(self, progress_ratio):
        """
        NEW: Update sparsity based on training progress
        """
        self.current_sparsity = self.initial_sparsity + progress_ratio * (self.fusion_sparsity - self.initial_sparsity)
        
        # Update sparsity for modality projections
        for projection in self.modality_projections.values():
            projection.update_sparsity(progress_ratio)
        
        # Update sparsity for fusion components
        if self.fusion_method == 'cross_attention' and hasattr(self, 'text2image'):
            self.text2image.update_sparsity(progress_ratio)
            self.image2text.update_sparsity(progress_ratio)
            
            if hasattr(self, 'cross_modal_align'):
                for align in self.cross_modal_align.values():
                    if hasattr(align, 'sparsity_target'):
                        align.sparsity_target = self.current_sparsity
            
            if hasattr(self.final_fusion, 'sparsity_target'):
                self.final_fusion.sparsity_target = self.current_sparsity
        
        elif self.fusion_method == 'concat':
            if hasattr(self.fusion_layer, 'sparsity_target'):
                self.fusion_layer.sparsity_target = self.current_sparsity
        
        else:  # Element-wise methods
            if hasattr(self.fusion_gate, 'sparsity_target'):
                self.fusion_gate.sparsity_target = self.current_sparsity
    
    def forward(self, inputs):
        """
        Forward pass for multimodal fusion
        
        Args:
            inputs: Dictionary of {modality_name: features} for each modality
                   Each modality's features should have shape [batch_size, ...]
        
        Returns:
            Fused multimodal features with shape [batch_size, output_dim]
        """
        # Check if required modalities are present
        for name in self.modality_projections.keys():
            if name not in inputs:
                raise ValueError(f"Required modality '{name}' not found in inputs")
        
        # Project each modality to common dimension with modality-specific quantization
        projected = {}
        for name, features in inputs.items():
            # Flatten if needed
            batch_size = features.size(0)
            if features.dim() > 2:
                features = features.view(batch_size, -1)
            
            # Project to common dimension
            projected[name] = self.modality_projections[name](features)
            
            # Apply modality-specific scaling
            scale = torch.clamp(self.modality_scales[name], min=0.5, max=2.0)
            projected[name] = projected[name] * scale
        
        # Apply fusion based on selected method
        if self.fusion_method == 'cross_attention' and 'text' in projected and 'image' in projected:
            # Cross-attention fusion (for two modalities)
            text_features = projected['text']
            image_features = projected['image']
            
            # Bidirectional cross-attention
            text_attended = self.text2image(text_features, image_features, image_features)
            image_attended = self.image2text(image_features, text_features, text_features)
            
            # NEW: Additional cross-modal alignment if available
            if hasattr(self, 'cross_modal_align'):
                text_attended = self.cross_modal_align['text'](text_attended)
                image_attended = self.cross_modal_align['image'](image_attended)
            
            # NEW: Apply L2 normalization for better similarity
            text_attended = F.normalize(text_attended, p=2, dim=1)
            image_attended = F.normalize(image_attended, p=2, dim=1)
            
            # Combine attended features
            combined = torch.cat([text_attended, image_attended], dim=1)
            
            fused = self.final_fusion(combined)
        
        elif self.fusion_method == 'concat':
            # Simple concatenation of projected features
            concatenated = torch.cat(list(projected.values()), dim=1)
            fused = self.fusion_layer(concatenated)
        
        else:  # Default to element-wise methods
            # Stack projected features for gating
            stacked = torch.cat(list(projected.values()), dim=1)
            gates = torch.sigmoid(self.fusion_gate(stacked))
            
            # Apply element-wise weighted sum
            fused = sum(gates[:, i:i+1] * feat for i, feat in enumerate(projected.values()))
        
        # Apply normalization and dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        # NEW: L2 normalize the final output for better contrastive learning
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused


# Example of usage
if __name__ == "__main__":
    # Create sample features for testing
    batch_size = 4
    image_dim = 512
    text_dim = 128
    output_dim = 256
    
    # Sample inputs
    image_features = torch.randn(batch_size, image_dim)
    text_features = torch.randn(batch_size, text_dim)
    
    # Create fusion module with cross-attention
    fusion_model = MultimodalFusion(
        input_dims={'image': image_dim, 'text': text_dim},
        output_dim=output_dim,
        fusion_method='cross_attention',
        use_rpb=True
    )
    
    # Forward pass
    inputs = {
        'image': image_features,
        'text': text_features
    }
    
    output = fusion_model(inputs)
    
    # Print output shape
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Fused output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")