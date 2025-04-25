import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing
from atq.quantizers import adaptive_ternary_quantization

class TernaryMultiheadAttention(nn.Module):
    """
    Multi-head attention with Adaptive Ternary Quantization
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_rpb=True, sparsity_target=0.3, 
                 attention_scale=None, critical_attention=False):
        super(TernaryMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rpb = use_rpb
        
        # NEW: Initial lower sparsity for better learning
        self.initial_sparsity = min(0.1, sparsity_target)
        self.target_sparsity = sparsity_target
        self.current_sparsity = self.initial_sparsity
        
        # NEW: Flag for critical attention paths
        self.critical_attention = critical_attention
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projection layers for query, key, value with ATQ
        precision_ratio = 0.2 if critical_attention else 0.05  # NEW: Higher precision for critical paths
        
        if use_rpb:
            self.q_proj = ResidualPrecisionBoostLinear(
                embed_dim, embed_dim, precision_ratio=precision_ratio, sparsity_target=self.initial_sparsity
            )
            self.k_proj = ResidualPrecisionBoostLinear(
                embed_dim, embed_dim, precision_ratio=precision_ratio, sparsity_target=self.initial_sparsity
            )
            self.v_proj = ResidualPrecisionBoostLinear(
                embed_dim, embed_dim, precision_ratio=precision_ratio, sparsity_target=self.initial_sparsity
            )
            self.out_proj = ResidualPrecisionBoostLinear(
                embed_dim, embed_dim, precision_ratio=precision_ratio*2, sparsity_target=self.initial_sparsity
            )
        else:
            self.q_proj = TernaryLinear(embed_dim, embed_dim)
            self.k_proj = TernaryLinear(embed_dim, embed_dim)
            self.v_proj = TernaryLinear(embed_dim, embed_dim)
            self.out_proj = TernaryLinear(embed_dim, embed_dim)
        
        # NEW: Custom attention scale to handle quantization effects
        self.attention_scale = attention_scale or (1.0 / math.sqrt(self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # NEW: Layer norm for better gradient flow in quantized networks
        self.pre_layer_norm = nn.LayerNorm(embed_dim)
        
    def update_sparsity(self, progress_ratio):
        """
        NEW: Gradually increase sparsity as training progresses
        """
        self.current_sparsity = self.initial_sparsity + progress_ratio * (self.target_sparsity - self.initial_sparsity)
        
        # Update sparsity for all projection layers
        if hasattr(self.q_proj, 'sparsity_target'):
            self.q_proj.sparsity_target = self.current_sparsity
            self.k_proj.sparsity_target = self.current_sparsity
            self.v_proj.sparsity_target = self.current_sparsity
            self.out_proj.sparsity_target = self.current_sparsity
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # NEW: Apply layer norm before quantized operations for better stability
        query = self.pre_layer_norm(query)
        
        batch_size = query.size(0)
        
        # Linear projections with ATQ
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Apply selective gradient routing for more stable training
        # NEW: Lower threshold for better gradient flow
        gradient_threshold = 0.01 if self.critical_attention else 0.05
        q = apply_selective_routing(q, threshold=gradient_threshold)
        k = apply_selective_routing(k, threshold=gradient_threshold)
        v = apply_selective_routing(v, threshold=gradient_threshold)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output, _ = self._scaled_dot_product_attention(
            q, k, v, attn_mask, key_padding_mask
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # NEW: Add residual connection for better gradient flow
        if self.critical_attention:
            output = output + 0.1 * query
            
        return output
    
    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_length]
            # scores: [batch_size, num_heads, seq_length_q, seq_length_k]
            
            # Get the dimensions
            batch_size, num_heads, seq_length_q, seq_length_k = scores.size()
            
            # Convert 1D mask to proper shape if needed
            if key_padding_mask.dim() == 1:
                # If mask is 1D, it might contain sequence lengths
                # Create proper mask
                lengths = key_padding_mask
                range_tensor = torch.arange(seq_length_k, device=scores.device).expand(batch_size, seq_length_k)
                key_padding_mask = range_tensor >= lengths.unsqueeze(1)
            
            # Ensure key_padding_mask has correct shape
            if key_padding_mask.dim() == 2:  # [batch_size, seq_length]
                # Reshape to [batch_size, 1, 1, seq_length_k]
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            elif key_padding_mask.dim() == 3:  # [batch_size, 1, seq_length]
                # Reshape to [batch_size, 1, 1, seq_length_k]
                key_padding_mask = key_padding_mask.unsqueeze(2)
            
            # Expand to match scores dimensions
            # [batch_size, 1, 1, seq_length_k] -> [batch_size, num_heads, seq_length_q, seq_length_k]
            key_padding_mask = key_padding_mask.expand(batch_size, num_heads, seq_length_q, seq_length_k)
            
            # Apply mask
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class TernaryTransformerLayer(nn.Module):
    """
    Transformer encoder layer with Adaptive Ternary Quantization
    """
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, 
                 use_rpb=True, sparsity_target=0.3, layer_idx=0):
        super(TernaryTransformerLayer, self).__init__()
        self.use_rpb = use_rpb
        
        # NEW: Initial lower sparsity for better learning
        self.initial_sparsity = min(0.1, sparsity_target)
        self.target_sparsity = sparsity_target
        self.current_sparsity = self.initial_sparsity
        self.layer_idx = layer_idx
        
        # NEW: Last layer is considered critical for representation
        is_critical = (layer_idx >= 0)  # All layers are important for small models
        
        # Multi-head self-attention with ATQ
        self.self_attn = TernaryMultiheadAttention(
            embed_dim, num_heads, dropout, use_rpb, self.initial_sparsity,
            critical_attention=is_critical
        )
        
        # Feed-forward network with ATQ
        # NEW: Increased precision ratio for better learning
        precision_ratio = 0.2 if is_critical else 0.05
        
        if use_rpb:
            self.linear1 = ResidualPrecisionBoostLinear(
                embed_dim, dim_feedforward, precision_ratio=precision_ratio, 
                sparsity_target=self.initial_sparsity
            )
            self.linear2 = ResidualPrecisionBoostLinear(
                dim_feedforward, embed_dim, precision_ratio=precision_ratio*2, 
                sparsity_target=self.initial_sparsity
            )
        else:
            self.linear1 = TernaryLinear(embed_dim, dim_feedforward)
            self.linear2 = TernaryLinear(dim_feedforward, embed_dim)
        
        # Layer norms are kept in full precision
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # NEW: Gating mechanism to control information flow through quantized network
        self.gate = nn.Parameter(torch.ones(1) * 0.8)
    
    def update_sparsity(self, progress_ratio):
        """
        NEW: Gradually increase sparsity as training progresses
        """
        self.current_sparsity = self.initial_sparsity + progress_ratio * (self.target_sparsity - self.initial_sparsity)
        
        # Update sparsity for attention
        self.self_attn.update_sparsity(progress_ratio)
        
        # Update sparsity for feed-forward layers
        if hasattr(self.linear1, 'sparsity_target'):
            self.linear1.sparsity_target = self.current_sparsity
            self.linear2.sparsity_target = self.current_sparsity
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # NEW: Pre-norm architecture for better training with quantization
        # Self-attention block with residual
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)
        
        # NEW: Gated residual connection
        gate_value = torch.sigmoid(self.gate)
        src = src + self.dropout1(src2) * gate_value
        
        # Feed-forward block with residual
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))  # NEW: GELU instead of ReLU
        src = src + self.dropout2(src2) * gate_value
        
        return src


class ATQTextEncoder(nn.Module):
    """
    Text encoder using Adaptive Ternary Quantization
    """
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, num_layers=4,  # NEW: Increased from 4 to 6 heads
                 dim_feedforward=512, dropout=0.1, use_rpb=True, 
                 sparsity_target=0.3, max_seq_length=256):
        super(ATQTextEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.use_rpb = use_rpb
        self.initial_sparsity = min(0.1, sparsity_target)  # NEW: Start with lower sparsity
        self.target_sparsity = sparsity_target
        self.current_sparsity = self.initial_sparsity
        
        # Word embeddings are kept in full precision for better representation
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # NEW: Layer norm after embedding for stable training
        self.embed_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_length, embed_dim)
        
        # NEW: Dropout after embedding for regularization
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer layers with ATQ
        self.layers = nn.ModuleList([
            TernaryTransformerLayer(
                embed_dim, num_heads, dim_feedforward, dropout, use_rpb, 
                sparsity_target=self.initial_sparsity,
                layer_idx=i
            ) for i in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # NEW: Enhanced attention pooling with higher precision
        if use_rpb:
            self.attention_pool = nn.Sequential(
                ResidualPrecisionBoostLinear(
                    embed_dim, embed_dim // 2, precision_ratio=0.2, 
                    sparsity_target=self.initial_sparsity
                ),
                nn.Tanh(),  # NEW: Tanh for better gradient flow
                ResidualPrecisionBoostLinear(
                    embed_dim // 2, 1, precision_ratio=0.2, 
                    sparsity_target=self.initial_sparsity
                ),
                nn.Softmax(dim=1)
            )
        else:
            self.attention_pool = nn.Sequential(
                TernaryLinear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                TernaryLinear(embed_dim // 2, 1),
                nn.Softmax(dim=1)
            )
        
        # NEW: Scaling factor for final embeddings
        self.scaling = nn.Parameter(torch.ones(1) * 4.0)
            
        # Initialize parameters
        self._init_parameters()
    
    def update_sparsity(self, progress_ratio):
        """
        NEW: Update sparsity across all components based on training progress
        """
        self.current_sparsity = self.initial_sparsity + progress_ratio * (self.target_sparsity - self.initial_sparsity)
        
        # Update all transformer layers
        for layer in self.layers:
            layer.update_sparsity(progress_ratio)
        
        # Update attention pooling if using RPB
        if self.use_rpb and hasattr(self.attention_pool[0], 'sparsity_target'):
            self.attention_pool[0].sparsity_target = self.current_sparsity
            self.attention_pool[2].sparsity_target = self.current_sparsity
    
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)
    
    def _init_parameters(self):
        """Initialize model parameters using better initialization for quantized networks"""
        # NEW: Better initialization for ternary quantization
        for p in self.parameters():
            if p.dim() > 1:
                # Initialize with smaller values for better ternary conversion
                nn.init.xavier_uniform_(p, gain=0.8)
                
        # Special initialization for embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass for text encoder
        
        Args:
            x: Input token indices of shape [batch_size, seq_length]
            src_key_padding_mask: Boolean mask for padding tokens (True for pad tokens)
                                Shape: [batch_size, seq_length]
                                Can also pass lengths instead and we'll create the mask
        
        Returns:
            Text features of shape [batch_size, embed_dim]
        """
        # Create padding mask from lengths if needed
        if src_key_padding_mask is not None:
            if isinstance(src_key_padding_mask, (list, tuple)):
                # Convert list/tuple of lengths to tensor
                src_key_padding_mask = torch.tensor(src_key_padding_mask, device=x.device)
            
            if not torch.is_tensor(src_key_padding_mask):
                # Convert to tensor if needed
                src_key_padding_mask = torch.tensor(src_key_padding_mask, device=x.device)
            else:
                # Ensure tensor is on the correct device
                src_key_padding_mask = src_key_padding_mask.to(x.device)
            
            if src_key_padding_mask.dim() == 1:
                # Assume src_key_padding_mask contains sequence lengths
                lengths = src_key_padding_mask
                batch_size = x.size(0)
                seq_length = x.size(1)
                
                # Create a mask where True means padding
                range_tensor = torch.arange(seq_length, device=x.device).expand(batch_size, seq_length)
                src_key_padding_mask = range_tensor >= lengths.unsqueeze(1)
        
        # Word embeddings
        x = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        
        # NEW: Apply layer norm to embeddings
        x = self.embed_norm(x)
        
        # Add positional encoding
        seq_length = x.size(1)
        x = x + self.positional_encoding[:, :seq_length, :]
        
        # NEW: Apply dropout after embedding
        x = self.embed_dropout(x)
        
        # Apply transformers layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Apply final normalization
        x = self.norm(x)  # [batch_size, seq_length, embed_dim]
        
        # Attention pooling to get fixed-size representation
        # Calculate attention weights
        attn_weights = self.attention_pool(x)  # [batch_size, seq_length, 1]
        
        # Mask out padding when applying attention weights
        if src_key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(src_key_padding_mask.unsqueeze(-1), float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights to get weighted sum
        text_features = torch.sum(x * attn_weights, dim=1)  # [batch_size, embed_dim]
        
        # NEW: Apply scaling factor to adjust magnitude (important for matching image features)
        scaling = torch.clamp(self.scaling, min=1.0, max=10.0)
        text_features = text_features * scaling
        
        return text_features
    
    def extract_features(self, x, src_key_padding_mask=None):
        """
        Extract features from text (alias for forward)
        """
        return self.forward(x, src_key_padding_mask)


# Example of usage
if __name__ == "__main__":
    # Create a small model for testing
    vocab_size = 10000
    batch_size = 4
    seq_length = 64
    
    # Sample input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Create model
    model = ATQTextEncoder(
        vocab_size=vocab_size,
        use_rpb=True,
        sparsity_target=0.3
    )
    
    # Forward pass
    output = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")