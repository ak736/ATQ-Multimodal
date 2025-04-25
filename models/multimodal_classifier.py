import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_classifier import ATQImageClassifier
from .text_encoder import ATQTextEncoder
from .fusion import MultimodalFusion
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    Enhanced image encoder using ATQ for multimodal retrieval
    """
    def __init__(self, embed_dim=256, use_rpb=True, sparsity_target=0.3, base_model='resnet18'):
        super(ImageEncoder, self).__init__()
        self.use_rpb = use_rpb
        
        # NEW: Gradual sparsity approach
        self.initial_sparsity = min(0.1, sparsity_target)
        self.target_sparsity = sparsity_target
        self.current_sparsity = self.initial_sparsity
        
        self.embed_dim = embed_dim
        
        # Load pretrained image model for feature extraction
        if base_model == 'resnet18':
            # Load pretrained ResNet model without final FC layer
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC
            self.feature_dim = 512
        elif base_model == 'resnet50':
            # Load pretrained ResNet50 for higher-capacity model
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # NEW: Add feature normalization
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        
        # NEW: Increase precision ratio for better representation
        precision_ratio = 0.2  # Higher precision for critical projection
        
        # Projection to embedding space with ATQ
        if use_rpb:
            self.projector = ResidualPrecisionBoostLinear(
                self.feature_dim, embed_dim, 
                precision_ratio=precision_ratio, 
                sparsity_target=self.initial_sparsity
            )
        else:
            self.projector = TernaryLinear(self.feature_dim, embed_dim)
        
        # NEW: Add non-linearity and normalization for robust features
        self.activation = nn.GELU()  # Better gradient flow than ReLU
        self.proj_norm = nn.LayerNorm(embed_dim)
        
        # NEW: Scaling factor for embeddings
        self.scaling = nn.Parameter(torch.ones(1) * 4.0)
        
        # L2 normalization for cosine similarity in embedding space
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)
    
    def update_sparsity(self, progress_ratio):
        """
        NEW: Update sparsity based on training progress
        """
        self.current_sparsity = self.initial_sparsity + progress_ratio * (self.target_sparsity - self.initial_sparsity)
        
        # Update sparsity for projection layer
        if hasattr(self.projector, 'sparsity_target'):
            self.projector.sparsity_target = self.current_sparsity
    
    def forward(self, x):
        # Extract features
        features = self.base_model(x)
        features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        
        # NEW: Apply normalization to raw features
        features = self.feature_norm(features)
        
        # Project to embedding space
        embeddings = self.projector(features)
        
        # NEW: Apply activation and normalization
        embeddings = self.activation(embeddings)
        embeddings = self.proj_norm(embeddings)
        
        # NEW: Apply scaling (important for matching text features)
        scaling = torch.clamp(self.scaling, min=1.0, max=10.0)
        embeddings = embeddings * scaling
        
        # L2 normalize embeddings for cosine similarity
        embeddings = self.l2_norm(embeddings)
        
        return embeddings


class ATQMultimodalRetrieval(nn.Module):
    """
    Multimodal retrieval model using Adaptive Ternary Quantization
    For image-text retrieval tasks
    """
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512,
                 vision_threshold=0.3, text_threshold=0.2, use_residual=True):
        super(ATQMultimodalRetrieval, self).__init__()
        self.use_rpb = use_residual
        self.embed_dim = embed_dim
        
        # NEW: Initial sparsity values - start lower
        self.initial_vision_sparsity = min(0.1, vision_threshold) 
        self.initial_text_sparsity = min(0.1, text_threshold)
        self.target_vision_sparsity = vision_threshold
        self.target_text_sparsity = text_threshold
        self.current_epoch = 0
        self.total_epochs = 20  # Assume 20 epochs by default
        
        # Create image encoder
        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim,
            use_rpb=use_residual,
            sparsity_target=self.initial_vision_sparsity
        )
        
        # Create text encoder with enhanced parameters
        self.text_encoder = ATQTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,  # Match image embedding dimension
            num_heads=8,          # NEW: Increased from 4 to 8
            num_layers=4,         # NEW: Increased from 2 to 4
            dim_feedforward=hidden_dim,
            use_rpb=use_residual,
            sparsity_target=self.initial_text_sparsity,
            max_seq_length=50
        )
        
        # NEW: Use cross-modal fusion instead of simple projection
        self.fusion = MultimodalFusion(
            input_dims={'image': embed_dim, 'text': embed_dim},
            output_dim=embed_dim,
            fusion_method='cross_attention',  # Use cross-attention for better alignment
            num_heads=4,
            use_rpb=use_residual
        )
        
        # Joint embedding space projection for text
        if use_residual:
            self.text_projector = ResidualPrecisionBoostLinear(
                embed_dim, embed_dim,
                precision_ratio=0.2,  # NEW: Increased from 0.1
                sparsity_target=self.initial_text_sparsity
            )
            
            # NEW: Additional projection for image features
            self.image_projector = ResidualPrecisionBoostLinear(
                embed_dim, embed_dim,
                precision_ratio=0.2,
                sparsity_target=self.initial_vision_sparsity
            )
        else:
            self.text_projector = TernaryLinear(embed_dim, embed_dim)
            self.image_projector = TernaryLinear(embed_dim, embed_dim)
        
        # L2 normalization for cosine similarity in embedding space
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)
        
        # Temperature parameter for similarity scaling (learnable with better initialization)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # NEW: Add LayerNorm for embedding normalization before computing similarity
        self.img_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
    
    def set_epoch(self, current_epoch, total_epochs):
        """
        NEW: Set the current epoch for progressive sparsity
        """
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Calculate progress ratio (0 to 1)
        progress_ratio = min(1.0, current_epoch / (total_epochs * 0.8))
        
        # Update sparsity across all components
        self._update_sparsity(progress_ratio)
    
    def _update_sparsity(self, progress_ratio):
        """
        NEW: Update sparsity across all components based on training progress
        """
        # Update image encoder sparsity
        self.image_encoder.update_sparsity(progress_ratio)
        
        # Update text encoder sparsity
        self.text_encoder.update_sparsity(progress_ratio)
        
        # Update fusion module sparsity
        self.fusion.update_sparsity(progress_ratio)
        
        # Update projector sparsity
        if hasattr(self.text_projector, 'sparsity_target'):
            current_text_sparsity = self.initial_text_sparsity + progress_ratio * (self.target_text_sparsity - self.initial_text_sparsity)
            current_vision_sparsity = self.initial_vision_sparsity + progress_ratio * (self.target_vision_sparsity - self.initial_vision_sparsity)
            
            self.text_projector.sparsity_target = current_text_sparsity
            self.image_projector.sparsity_target = current_vision_sparsity
    
    def encode_image(self, image):
        """Encode image to embedding space"""
        return self.image_encoder(image)
    
    def encode_text(self, text, text_lengths=None):
        """Encode text to embedding space"""
        # Pass text_lengths directly to text_encoder which will create the proper mask
        text_features = self.text_encoder(text, text_lengths)
        
        # NEW: Apply additional projection for better representation
        text_embeddings = self.text_projector(text_features)
        
        # NEW: Apply layer normalization
        text_embeddings = self.text_norm(text_embeddings)
        
        return self.l2_norm(text_embeddings)
    
    def forward(self, image, text, text_lengths=None, return_embeddings=False, return_fused=False):
        """
        Forward pass for computing similarity between image and text
        
        Args:
            image: Image tensor of shape [batch_size, channels, height, width]
            text: Text tensor of shape [batch_size, seq_length]
            text_lengths: Optional lengths of text sequences
            return_embeddings: Whether to return embeddings instead of similarity scores
            return_fused: NEW: Whether to return fused embeddings
        
        Returns:
            Similarity scores or embeddings depending on return_embeddings
        """
        # Encode image and text to embedding space
        image_embeddings = self.encode_image(image)
        text_embeddings = self.encode_text(text, text_lengths)
        
        if return_embeddings:
            return image_embeddings, text_embeddings
        
        # NEW: Apply cross-modal fusion if requested
        if return_fused:
            fused_embeddings = self.fusion({
                'image': image_embeddings,
                'text': text_embeddings
            })
            return fused_embeddings
        
        # NEW: Apply additional projection for image embeddings
        image_embeddings = self.image_projector(image_embeddings)
        image_embeddings = self.img_norm(image_embeddings)
        image_embeddings = self.l2_norm(image_embeddings)
        
        # Compute similarity scores
        # For each image, compute similarity with all texts in batch
        # sim[i, j] = similarity between image i and text j
        similarity = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        return similarity
    
    def get_model_size_info(self):
        """
        Calculate model size information
        
        Returns:
            Dictionary with parameter counts and estimated memory usage
        """
        # Count parameters
        image_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        text_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        text_proj_params = sum(p.numel() for p in self.text_projector.parameters() if p.requires_grad)
        image_proj_params = sum(p.numel() for p in self.image_projector.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        
        total_params = image_params + text_params + text_proj_params + image_proj_params + fusion_params
        
        # Estimate memory usage (rough approximation)
        # For full precision: 4 bytes per parameter
        # For ternary: 2 bits per parameter (theoretical with bit-packing)
        if self.use_rpb:
            # Assume 75% of parameters can be ternarized with RPB
            memory_usage_bytes = (total_params * 0.75 * 2/8) + (total_params * 0.25 * 4)
        else:
            # More aggressive ternarization without RPB
            memory_usage_bytes = (total_params * 0.9 * 2/8) + (total_params * 0.1 * 4)
        
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'image_encoder_parameters': image_params,
            'text_encoder_parameters': text_params,
            'text_projector_parameters': text_proj_params,
            'image_projector_parameters': image_proj_params,
            'fusion_parameters': fusion_params,
            'estimated_memory_usage_MB': memory_usage_mb
        }
    
    def modality_dropout(self, rate=0.1):
        """
        Apply modality dropout during training for robustness
        
        Args:
            rate: Probability of dropping a modality
        """
        if self.training:
            self.drop_image = torch.rand(1).item() < rate
            self.drop_text = torch.rand(1).item() < rate
        else:
            self.drop_image = False
            self.drop_text = False


# Original class for backward compatibility
class ATQMultimodalClassifier(nn.Module):
    """
    Original multimodal classifier for backward compatibility
    """
    def __init__(self, num_classes=10, vocab_size=10000, embed_dim=128, hidden_dim=256,
                fusion_method='cross_attention', vision_threshold=0.05, text_threshold=0.05,
                fusion_threshold=0.05, use_residual=True, residual_scale=0.1):
        super(ATQMultimodalClassifier, self).__init__()
        self.use_rpb = use_residual
        self.residual_scale = residual_scale
        
        # NEW: Gradual sparsity approach
        self.initial_vision_sparsity = min(0.01, vision_threshold)
        self.initial_text_sparsity = min(0.01, text_threshold)
        self.initial_fusion_sparsity = min(0.01, fusion_threshold)
        self.target_vision_sparsity = vision_threshold
        self.target_text_sparsity = text_threshold
        self.target_fusion_sparsity = fusion_threshold
        self.current_epoch = 0
        self.total_epochs = 20  # Assume 20 epochs by default
        
        # Create image encoder (based on your existing ATQImageClassifier)
        self.image_encoder = ATQImageClassifier(
            num_classes=num_classes,  # Note: We won't use the classification head
            use_rpb=use_residual,
            sparsity_target=self.initial_vision_sparsity
        )
        
        # Create text encoder
        self.text_encoder = ATQTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            dim_feedforward=hidden_dim,
            use_rpb=use_residual,
            sparsity_target=self.initial_text_sparsity
        )
        
        # Get feature dimensions from each encoder
        # For Fashion-MNIST (28x28), after 2 max-pooling layers (factor of 4 reduction)
        # with 64 channels, the feature size would be 64 * 7 * 7 = 3136
        image_feature_dim = 64 * 7 * 7  # = 3136
        text_feature_dim = embed_dim   # As set in constructor
        
        # Create fusion module
        self.fusion = MultimodalFusion(
            input_dims={'image': image_feature_dim, 'text': text_feature_dim},
            output_dim=hidden_dim,
            fusion_method=fusion_method,
            use_rpb=use_residual
        )
        
        # Classification head with ATQ
        if use_residual:
            self.classifier = nn.Sequential(
                ResidualPrecisionBoostLinear(
                    hidden_dim, hidden_dim // 2,
                    precision_ratio=0.2,  # NEW: Increased precision ratio
                    sparsity_target=self.initial_fusion_sparsity
                ),
                nn.GELU(),  # NEW: GELU instead of ReLU
                nn.Dropout(0.2),
                ResidualPrecisionBoostLinear(
                    hidden_dim // 2, num_classes,
                    precision_ratio=0.2,  # NEW: Increased precision ratio
                    sparsity_target=self.initial_fusion_sparsity
                )
            )
        else:
            self.classifier = nn.Sequential(
                TernaryLinear(hidden_dim, hidden_dim // 2),
                nn.GELU(),  # NEW: GELU instead of ReLU
                nn.Dropout(0.2),
                TernaryLinear(hidden_dim // 2, num_classes)
            )
        
        # Optional: Mixed precision path (full precision classifier for critical tasks)
        self.full_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),  # NEW: GELU instead of ReLU
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Mixing parameter for quantized vs full-precision path
        self.mix_ratio = nn.Parameter(torch.tensor(0.8))  # Start with higher weight on quantized path
        
        # For modality dropout
        self.drop_image = False
        self.drop_text = False
    
    def set_epoch(self, current_epoch, total_epochs):
        """
        NEW: Set the current epoch for progressive sparsity
        """
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Calculate progress ratio (0 to 1)
        progress_ratio = min(1.0, current_epoch / (total_epochs * 0.8))
        
        # Update sparsity across all components
        self._update_sparsity(progress_ratio)
    
    def _update_sparsity(self, progress_ratio):
        """
        NEW: Update sparsity across all components based on training progress
        """
        # Calculate current sparsity values
        vision_sparsity = self.initial_vision_sparsity + progress_ratio * (self.target_vision_sparsity - self.initial_vision_sparsity)
        text_sparsity = self.initial_text_sparsity + progress_ratio * (self.target_text_sparsity - self.initial_text_sparsity)
        fusion_sparsity = self.initial_fusion_sparsity + progress_ratio * (self.target_fusion_sparsity - self.initial_fusion_sparsity)
        
        # Update sparsity in image encoder
        if hasattr(self.image_encoder, 'update_sparsity'):
            self.image_encoder.update_sparsity(progress_ratio)
        elif hasattr(self.image_encoder, 'sparsity_target'):
            self.image_encoder.sparsity_target = vision_sparsity
        
        # Update sparsity in text encoder
        if hasattr(self.text_encoder, 'update_sparsity'):
            self.text_encoder.update_sparsity(progress_ratio)
        elif hasattr(self.text_encoder, 'sparsity_target'):
            self.text_encoder.sparsity_target = text_sparsity
        
        # Update sparsity in fusion module
        if hasattr(self.fusion, 'update_sparsity'):
            self.fusion.update_sparsity(progress_ratio)
        
        # Update sparsity in classifier
        if self.use_rpb:
            if hasattr(self.classifier[0], 'sparsity_target'):
                self.classifier[0].sparsity_target = fusion_sparsity
            if hasattr(self.classifier[3], 'sparsity_target'):
                self.classifier[3].sparsity_target = fusion_sparsity
    
    def forward(self, image, text, text_padding_mask=None):
        """Forward pass (kept for backward compatibility)"""
        # Get batch size
        batch_size = image.size(0)
        
        # Extract image features
        image_features = self.image_encoder.extract_features(image)
        
        # Apply modality dropout during training if enabled
        if self.training and self.drop_image:
            image_features = torch.zeros_like(image_features)
        
        # Flatten image features if needed
        if len(image_features.shape) > 2:
            image_features = image_features.reshape(batch_size, -1)
        
        # Extract text features
        text_features = self.text_encoder(text, text_padding_mask)
        
        # Apply modality dropout during training if enabled
        if self.training and self.drop_text:
            text_features = torch.zeros_like(text_features)
        
        # Prepare multimodal inputs for fusion
        multimodal_inputs = {
            'image': image_features,
            'text': text_features
        }
        
        # Fuse modalities
        fused_features = self.fusion(multimodal_inputs)
        
        # Apply selective gradient routing for stable training
        # NEW: Lower threshold for better gradient flow
        fused_routed = apply_selective_routing(fused_features, threshold=0.01)
        
        # Classification with both paths
        quant_logits = self.classifier(fused_routed)
        full_logits = self.full_classifier(fused_features)
        
        # Mix quantized and full-precision paths for better accuracy
        mix = torch.sigmoid(self.mix_ratio)
        logits = mix * quant_logits + (1 - mix) * full_logits
        
        return logits
    
    def get_model_size_info(self):
        """
        Calculate model size information
        
        Returns:
            Dictionary with parameter counts and estimated memory usage
        """
        # Count parameters
        image_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        text_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        full_classifier_params = sum(p.numel() for p in self.full_classifier.parameters() if p.requires_grad)
        
        total_params = image_params + text_params + fusion_params + classifier_params + full_classifier_params
        
        # Estimate memory usage (rough approximation)
        # For full precision: 4 bytes per parameter
        # For ternary: 2 bits per parameter (theoretical with bit-packing)
        if self.use_rpb:
            # Assume 75% of parameters can be ternarized with RPB
            memory_usage_bytes = (total_params * 0.75 * 2/8) + (total_params * 0.25 * 4)
        else:
            # More aggressive ternarization without RPB
            memory_usage_bytes = (total_params * 0.9 * 2/8) + (total_params * 0.1 * 4)
        
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'image_encoder_parameters': image_params,
            'text_encoder_parameters': text_params,
            'fusion_parameters': fusion_params,
            'classifier_parameters': classifier_params,
            'full_precision_classifier_parameters': full_classifier_params,
            'estimated_memory_usage_MB': memory_usage_mb
        }
    
    def modality_dropout(self, rate=0.1):
        """
        Apply modality dropout during training for robustness
        
        Args:
            rate: Probability of dropping a modality
        """
        if self.training:
            self.drop_image = torch.rand(1).item() < rate
            self.drop_text = torch.rand(1).item() < rate
        else:
            self.drop_image = False
            self.drop_text = False


# Alias for backward compatibility 
class MultimodalATQ(ATQMultimodalClassifier):
    """Alias for ATQMultimodalClassifier"""
    pass