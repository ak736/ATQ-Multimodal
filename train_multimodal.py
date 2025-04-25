import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import math
from tqdm import tqdm

from models.multimodal_classifier import ATQMultimodalRetrieval
from data.multimodal_data import prepare_flickr8k_dataloaders, visualize_flickr8k_samples
from utils.metrics import measure_model_memory, measure_inference_time, count_parameters
from atq.mixed_precision_atq import GradualQuantizationScheduler, MixedPrecisionATQ
from utils.enhanced_contrastive import HardNegativeMiningInfoNCE, ContrastiveLearningManager

# Enable MPS fallback to improve performance for operations not supported by MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def compute_retrieval_metrics(similarity, topk=[1, 5, 10]):
    """
    Compute retrieval metrics: Recall@K
    
    Args:
        similarity: Similarity matrix of shape [num_images, num_texts]
                   similarity[i, j] = similarity between image i and text j
        topk: List of K values for Recall@K
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # For full evaluation, diagonal elements are positive pairs
    # Image-to-text retrieval
    i2t_ranks = []
    for i in range(similarity.size(0)):
        # Get the similarity scores for this image with all texts
        sim_scores = similarity[i].clone()
        
        # Target is the corresponding text (diagonal element)
        target_index = i
        
        # Slightly lower the target score to handle equal scores
        if target_index < len(sim_scores):
            target_score = sim_scores[target_index].clone()
            sim_scores[target_index] -= 1e-6
            
            # Calculate rank with greater numerical stability
            # Count how many texts have a higher score than the target
            rank = (sim_scores >= target_score).sum().item()
            i2t_ranks.append(rank)
    
    # Text-to-image retrieval
    t2i_ranks = []
    for i in range(similarity.size(1)):
        # Get the similarity scores for this text with all images
        sim_scores = similarity[:, i].clone()
        
        # Target is the corresponding image (diagonal element)
        target_index = i
        
        # Slightly lower the target score to handle equal scores
        if target_index < len(sim_scores):
            target_score = sim_scores[target_index].clone()
            sim_scores[target_index] -= 1e-6
            
            # Calculate rank with greater numerical stability
            # Count how many images have a higher score than the target
            rank = (sim_scores >= target_score).sum().item()
            t2i_ranks.append(rank)
    
    # Calculate Recall@K for both directions
    for k in topk:
        if len(i2t_ranks) > 0:
            metrics[f'image_to_text_R@{k}'] = 100 * sum([1 if r <= k else 0 for r in i2t_ranks]) / len(i2t_ranks)
        else:
            metrics[f'image_to_text_R@{k}'] = 0.0
            
        if len(t2i_ranks) > 0:
            metrics[f'text_to_image_R@{k}'] = 100 * sum([1 if r <= k else 0 for r in t2i_ranks]) / len(t2i_ranks)
        else:
            metrics[f'text_to_image_R@{k}'] = 0.0
            
        metrics[f'mean_R@{k}'] = (metrics[f'image_to_text_R@{k}'] + metrics[f'text_to_image_R@{k}']) / 2
    
    return metrics


def create_baseline_model(vocab_size, embed_dim, hidden_dim):
    """
    Create full-precision baseline model for retrieval
    """
    import torchvision.models as models
    
    class BaselineRetrievalModel(nn.Module):
        def __init__(self):
            super(BaselineRetrievalModel, self).__init__()
            
            # Image encoder
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.image_encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC
            self.image_projector = nn.Sequential(
                nn.Linear(512, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim)
            )
            
            # Text encoder
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.text_encoder = nn.GRU(
                embed_dim, hidden_dim, 
                batch_first=True, 
                bidirectional=True
            )
            self.text_projector = nn.Sequential(
                nn.Linear(hidden_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim)
            )
            
            # L2 normalization for cosine similarity
            self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)
            
            # Temperature parameter
            self.temperature = nn.Parameter(torch.tensor(0.07))
        
        def encode_image(self, image):
            features = self.image_encoder(image)
            features = features.squeeze(-1).squeeze(-1)
            embeddings = self.image_projector(features)
            return self.l2_norm(embeddings)
        
        def encode_text(self, text, text_lengths=None):
            batch_size = text.size(0)
            
            # Get device
            device = text.device
            
            # Create a packed sequence for variable-length sequences
            if text_lengths is not None and text_lengths.sum() > 0:
                # Sort sequences by length for PackedSequence
                sorted_lengths, indices = torch.sort(text_lengths, descending=True)
                sorted_text = text[indices]
                
                # Handle empty sequences
                max_len = int(sorted_lengths[0].item())
                if max_len == 0:
                    max_len = 1
                
                # Embed tokens
                embedded = self.embedding(sorted_text[:, :max_len])
                
                # Create packed sequence
                packed_embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
                )
                
                # Process with GRU
                _, hidden = self.text_encoder(packed_embedded)
                
                # Restore original order
                _, reverse_indices = torch.sort(indices)
                hidden = hidden.view(2, 2, batch_size, hidden_dim // 2)[-1]
                hidden = hidden.transpose(0, 1).reshape(batch_size, hidden_dim)
                hidden = hidden[reverse_indices]
            else:
                # Fallback for problematic lengths
                embedded = self.embedding(text)
                output, hidden = self.text_encoder(embedded)
                hidden = torch.cat([hidden[0], hidden[1]], dim=1)
            
            # Project to embedding space
            embeddings = self.text_projector(hidden)
            return self.l2_norm(embeddings)
        
        def forward(self, image, text, text_lengths=None, return_embeddings=False):
            # Encode image and text
            image_embeddings = self.encode_image(image)
            text_embeddings = self.encode_text(text, text_lengths)
            
            if return_embeddings:
                return image_embeddings, text_embeddings
            
            # Compute similarity
            similarity = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
            return similarity
    
    return BaselineRetrievalModel()


def evaluate_model(model, dataloader, device, topk=[1, 5, 10]):
    """
    Evaluate retrieval model
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device for computation
        topk: List of K values for Recall@K
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Store all embeddings instead of similarities
    all_image_embeddings = []
    all_text_embeddings = []
    
    with torch.no_grad():
        for batch_idx, (images, captions, lengths) in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move all data to the correct device
            images = images.to(device)
            captions = captions.to(device)
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=device)
            else:
                lengths = lengths.to(device)
            
            # Get embeddings instead of similarity
            image_embeds, text_embeds = model(images, captions, lengths, return_embeddings=True)
            all_image_embeddings.append(image_embeds.cpu())
            all_text_embeddings.append(text_embeds.cpu())
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Compute the full similarity matrix
    similarity = torch.matmul(all_image_embeddings, all_text_embeddings.t())
    
    # Compute retrieval metrics
    metrics = compute_retrieval_metrics(similarity, topk=topk)
    
    return metrics


def train_retrieval(args):
    """Train a multimodal ATQ model for image-text retrieval"""
    # Set device
    if torch.backends.mps.is_available() and args.device == "mps":
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset
    print("Preparing Flickr8k dataset...")
    train_loader, val_loader, test_loader, vocab_size, word_to_idx = prepare_flickr8k_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_length=args.max_seq_length,
        tokenize_captions=True,
        num_workers=args.num_workers
    )
    
    # Create idx_to_word mapping for visualization
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Visualize samples
    visualize_flickr8k_samples(train_loader, num_samples=5, idx_to_word=idx_to_word)
    
    # Create ATQ retrieval model
    print("Creating ATQ retrieval model...")
    model = ATQMultimodalRetrieval(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vision_threshold=args.vision_sparsity,
        text_threshold=args.text_sparsity,
        use_residual=args.use_residual
    )
    model = model.to(device)
    
    # Initialize model properly
    def initialize_model(model):
        """Initialize model with better values for quantization"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    # Embedding needs special initialization
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif len(param.shape) >= 2:
                    # Initialize weights matrices with scaled Xavier
                    nn.init.xavier_uniform_(param, gain=0.8)
                else:
                    # Initialize 1D weights with small normal values
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                # Initialize biases as zeros
                nn.init.zeros_(param)
                
    if args.reinit_model:
        print("Reinitializing model weights...")
        initialize_model(model)
    
    # Create baseline model if needed
    if args.train_baseline:
        print("Creating baseline retrieval model...")
        baseline_model = create_baseline_model(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim
        )
        baseline_model = baseline_model.to(device)
    
    # Print model information
    model_info = model.get_model_size_info()
    print("\nModel information:")
    for key, value in model_info.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value:.2f}")
    
    # Create enhanced contrastive loss
    criterion = HardNegativeMiningInfoNCE(
        temperature=0.07,
        lambda_reg=args.contrastive_reg,
        hard_negative_weight=0.5,
        temperature_schedule=True
    )
    
    # Create contrastive learning manager
    cl_manager = ContrastiveLearningManager(
        model=model,
        criterion=criterion,
        similarity_threshold=0.7
    )
    
    # Create gradual quantization scheduler
    if args.gradual_quant:
        scheduler = GradualQuantizationScheduler(
            model, 
            args.epochs, 
            vision_sparsity=args.vision_sparsity, 
            text_sparsity=args.text_sparsity, 
            warmup_epochs=args.warmup_epochs,
            verbose=args.verbose
        )
    
    # Create optimizer with appropriate learning rate
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98)
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=0.9
        )
    else:  # Default to Adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98)
        )
    
    # Baseline optimizer
    if args.train_baseline:
        baseline_optimizer = optim.AdamW(
            baseline_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)  # 10% warmup
    
    def get_lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
    
    # Training metrics
    best_val_r1 = 0.0
    train_losses = []
    val_metrics_history = []
    
    # Set up mixed precision if requested
    if args.use_amp and device.type in ['cuda', 'mps']:
        use_amp = True
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    else:
        use_amp = False
        scaler = None
    
    # Create EMA model if requested
    if args.use_ema:
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_decay = 0.999
        
        # Function to update EMA model
        def update_ema_model(model, ema_model, decay):
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Update contrastive learning components for current epoch
        criterion.set_epoch(epoch, args.epochs)
        cl_manager.set_epoch(epoch, args.epochs)
        
        # Update model's quantization parameters
        if args.gradual_quant:
            scheduler.step(epoch)
        else:
            # Legacy mode - use model's built-in epoch tracking if available
            if hasattr(model, 'set_epoch'):
                model.set_epoch(epoch, args.epochs)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        start_time = time.time()
        for batch_idx, (images, captions, lengths) in enumerate(progress_bar):
            images, captions = images.to(device), captions.to(device)
            # Ensure lengths is a tensor on the correct device
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=device)
            elif lengths.device != device:
                lengths = lengths.to(device)
            
            # Apply modality dropout if enabled
            if args.modality_dropout > 0:
                model.modality_dropout(args.modality_dropout)
            
            # Memory optimization: Clear GPU cache periodically
            if device.type == 'mps' and batch_idx % 10 == 0:
                torch.mps.empty_cache()
            
            # Train baseline model if needed
            if args.train_baseline:
                baseline_model.train()
                baseline_optimizer.zero_grad()
                baseline_similarity = baseline_model(images, captions, lengths)
                baseline_loss = criterion(
                    *baseline_model(images, captions, lengths, return_embeddings=True)
                )
                baseline_loss.backward()
                baseline_optimizer.step()
                
                # Clear memory after baseline step
                if device.type in ['mps', 'cuda']:
                    torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()
            
            # Train ATQ model
            optimizer.zero_grad()
            
            try:
                # Automatic mixed precision training
                if use_amp:
                    with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.autocast('cpu'):
                        # Get embeddings
                        image_embeddings, text_embeddings = model(
                            images, captions, lengths, return_embeddings=True
                        )
                        
                        # Calculate contrastive loss with advanced methods
                        loss = cl_manager.compute_loss(image_embeddings, text_embeddings)
                        
                        # Knowledge distillation if enabled and baseline is available
                        if args.distill and args.train_baseline:
                            with torch.no_grad():
                                baseline_img_embed, baseline_txt_embed = baseline_model(
                                    images, captions, lengths, return_embeddings=True
                                )
                            
                            # KL divergence loss for each modality
                            temp = 3.0  # Temperature for distillation
                            img_sim = torch.matmul(image_embeddings, baseline_img_embed.t()) / temp
                            txt_sim = torch.matmul(text_embeddings, baseline_txt_embed.t()) / temp
                            
                            distill_img_loss = F.kl_div(
                                F.log_softmax(img_sim, dim=1),
                                F.softmax(img_sim.detach(), dim=1),
                                reduction='batchmean'
                            ) * (temp ** 2)
                            
                            distill_txt_loss = F.kl_div(
                                F.log_softmax(txt_sim, dim=1),
                                F.softmax(txt_sim.detach(), dim=1),
                                reduction='batchmean'
                            ) * (temp ** 2)
                            
                            distill_loss = (distill_img_loss + distill_txt_loss) / 2
                            
                            # Combined loss
                            loss = (1 - args.distill_weight) * loss + args.distill_weight * distill_loss
                    
                    # Backward and optimize with scaler if using CUDA
                    if device.type == 'cuda':
                        scaler.scale(loss).backward()
                        if args.clip_grad:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if args.clip_grad:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                else:
                    # Standard training without mixed precision
                    # Get embeddings
                    image_embeddings, text_embeddings = model(
                        images, captions, lengths, return_embeddings=True
                    )
                    
                    # Calculate contrastive loss with advanced methods
                    loss = cl_manager.compute_loss(image_embeddings, text_embeddings)
                    
                    # Knowledge distillation if enabled and baseline is available
                    if args.distill and args.train_baseline:
                        with torch.no_grad():
                            baseline_img_embed, baseline_txt_embed = baseline_model(
                                images, captions, lengths, return_embeddings=True
                            )
                        
                        # KL divergence loss for each modality
                        temp = 3.0  # Temperature for distillation
                        img_sim = torch.matmul(image_embeddings, baseline_img_embed.t()) / temp
                        txt_sim = torch.matmul(text_embeddings, baseline_txt_embed.t()) / temp
                        
                        distill_img_loss = F.kl_div(
                            F.log_softmax(img_sim, dim=1),
                            F.softmax(img_sim.detach(), dim=1),
                            reduction='batchmean'
                        ) * (temp ** 2)
                        
                        distill_txt_loss = F.kl_div(
                            F.log_softmax(txt_sim, dim=1),
                            F.softmax(txt_sim.detach(), dim=1),
                            reduction='batchmean'
                        ) * (temp ** 2)
                        
                        distill_loss = (distill_img_loss + distill_txt_loss) / 2
                        
                        # Combined loss
                        loss = (1 - args.distill_weight) * loss + args.distill_weight * distill_loss
                    
                    # Backward and optimize
                    loss.backward()
                    
                    # Gradient clipping for stability
                    if args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                
                # Update EMA model if enabled
                if args.use_ema:
                    update_ema_model(model, ema_model, ema_decay)
                
                # Track metrics
                train_loss += loss.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e) or "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    print(f"\nError in batch {batch_idx}: {e}")
                    # Try to recover
                    if device.type in ['cuda', 'mps']:
                        torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()
                    
                    # Skip this batch
                    optimizer.zero_grad()
                    continue
                else:
                    # Re-raise error if it's not memory related
                    raise e
            
            # Update learning rate
            scheduler.step()
            
            # Clear memory periodically
            if batch_idx % 5 == 0 and device.type in ['mps', 'cuda']:
                torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        print(f"\nValidating...")
        # Use EMA model for evaluation if enabled
        eval_model = ema_model if args.use_ema else model
        eval_model.eval()
        val_metrics = evaluate_model(eval_model, val_loader, device, topk=[1, 5, 10])
        val_metrics_history.append(val_metrics)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - {epoch_time:.1f}s:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation R@1: {val_metrics['mean_R@1']:.2f}%")
        print(f"  Validation R@5: {val_metrics['mean_R@5']:.2f}%")
        print(f"  Validation R@10: {val_metrics['mean_R@10']:.2f}%")
        
        # Save best model
        if val_metrics['mean_R@1'] > best_val_r1:
            best_val_r1 = val_metrics['mean_R@1']
            print(f"  New best model with validation R@1: {best_val_r1:.2f}%")
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            
            # Also save EMA model if enabled
            if args.use_ema:
                torch.save(ema_model.state_dict(), os.path.join(args.output_dir, "best_ema_model.pth"))
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_r1': best_val_r1,
                'train_losses': train_losses,
                'val_metrics': val_metrics_history,
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Memory cleanup at end of epoch
        if device.type in ['cuda', 'mps']:
            torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_metrics': val_metrics_history
    }
    
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        # Convert values to JSON serializable format
        history_serializable = {
            'train_losses': [float(x) for x in train_losses],
            'val_metrics': [{k: float(v) for k, v in metrics.items()} for metrics in val_metrics_history]
        }
        json.dump(history_serializable, f, indent=4)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot retrieval metrics
    plt.subplot(2, 2, 2)
    plt.plot([metrics['mean_R@1'] for metrics in val_metrics_history], label='R@1')
    plt.plot([metrics['mean_R@5'] for metrics in val_metrics_history], label='R@5')
    plt.plot([metrics['mean_R@10'] for metrics in val_metrics_history], label='R@10')
    plt.title('Validation Retrieval Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Recall (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot image-to-text vs text-to-image
    plt.subplot(2, 2, 3)
    plt.plot([metrics['image_to_text_R@1'] for metrics in val_metrics_history], label='Image→Text')
    plt.plot([metrics['text_to_image_R@1'] for metrics in val_metrics_history], label='Text→Image')
    plt.title('R@1 by Direction')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@1 (%)')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    plt.close()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    # Choose the best model for final evaluation
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print(f"Best model not found at {best_model_path}, using final model instead")
        # Check if we have a checkpoint, otherwise use current model
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoint_files:
            # Load the latest checkpoint
            latest_checkpoint = sorted(checkpoint_files)[-1]
            checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
            print(f"Loading from latest checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoints found, using current model state")
    
    # Use EMA model for final evaluation if enabled and available
    if args.use_ema and os.path.exists(os.path.join(args.output_dir, "best_ema_model.pth")):
        ema_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_ema_model.pth")))
        print("Using EMA model for final evaluation")
        eval_model = ema_model
    else:
        eval_model = model
    
    # Set model to eval mode
    eval_model.eval()
    
    # Test on the test set
    test_metrics = evaluate_model(eval_model, test_loader, device, topk=[1, 5, 10])
    
    # Measure inference time
    print("\nMeasuring inference times...")
    sample_image = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    sample_text = torch.ones(1, args.max_seq_length, dtype=torch.long).to(device)
    
    # Measure inference time for ATQ model
    atq_time = measure_inference_time(model, [sample_image, sample_text])
    
    # Measure inference time for baseline model if available
    if args.train_baseline:
        baseline_time = measure_inference_time(baseline_model, [sample_image, sample_text])
    else:
        baseline_time = 0
    
    # Final report
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    print(f"Best validation R@1: {best_val_r1:.2f}%")
    print(f"Test set metrics:")
    print(f"  R@1: {test_metrics['mean_R@1']:.2f}%")
    print(f"  R@5: {test_metrics['mean_R@5']:.2f}%")
    print(f"  R@10: {test_metrics['mean_R@10']:.2f}%")
    print(f"  Image→Text R@1: {test_metrics['image_to_text_R@1']:.2f}%")
    print(f"  Text→Image R@1: {test_metrics['text_to_image_R@1']:.2f}%")
    
    print(f"\nEfficiency metrics:")
    print(f"  ATQ inference time: {atq_time:.2f} ms per sample")
    if args.train_baseline:
        print(f"  Baseline inference time: {baseline_time:.2f} ms per sample")
        print(f"  Speed ratio: {baseline_time/atq_time:.2f}x")
    
    print(f"  Model size: {model_info['estimated_memory_usage_MB']:.2f} MB (estimated with ternarization)")
    
    # Save final report
    report = {
        'best_val_r1': best_val_r1,
        'test_metrics': test_metrics,
        'atq_inference_time_ms': atq_time,
        'baseline_inference_time_ms': baseline_time if args.train_baseline else None,
        'speed_ratio': baseline_time/atq_time if args.train_baseline and baseline_time > 0 and atq_time > 0 else None,
        'model_size_mb': model_info['estimated_memory_usage_MB'],
        'parameters': model_info['total_parameters'],
        'training_args': vars(args)
    }
    
    # Convert to JSON serializable format
    report_serializable = {
        'best_val_r1': float(best_val_r1),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'atq_inference_time_ms': float(atq_time),
        'baseline_inference_time_ms': float(baseline_time) if args.train_baseline else None,
        'speed_ratio': float(baseline_time/atq_time) if args.train_baseline and baseline_time > 0 and atq_time > 0 else None,
        'model_size_mb': float(model_info['estimated_memory_usage_MB']),
        'parameters': model_info['total_parameters'],
        'training_args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, "final_report.json"), "w") as f:
        json.dump(report_serializable, f, indent=4)
    
    return model, history, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ATQ model for image-text retrieval")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--output_dir", type=str, default="./outputs/retrieval", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    
    # Dataset settings
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=50, help="Maximum sequence length for text")
    parser.add_argument("--image_size", type=int, default=160, help="Image size for resizing")
    
    # Model settings
    parser.add_argument("--embed_dim", type=int, default=192, help="Embedding dimension for joint space")
    parser.add_argument("--hidden_dim", type=int, default=384, help="Hidden dimension for encoders")
    parser.add_argument("--vision_sparsity", type=float, default=0.3, help="Sparsity target for vision encoder")
    parser.add_argument("--text_sparsity", type=float, default=0.2, help="Sparsity target for text encoder")
    parser.add_argument("--use_residual", action="store_true", help="Use residual precision boosting")
    parser.add_argument("--reinit_model", action="store_true", help="Reinitialize model weights")
    
    # Enhanced ATQ settings
    parser.add_argument("--gradual_quant", action="store_true", help="Use gradual quantization schedule")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs for quantization")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer")
    parser.add_argument("--clip_grad", action="store_true", help="Apply gradient clipping")
    parser.add_argument("--modality_dropout", type=float, default=0.1, help="Probability of dropping a modality")
    parser.add_argument("--checkpoint_freq", type=int, default=2, help="Checkpoint save frequency (epochs)")
    parser.add_argument("--contrastive_reg", type=float, default=0.02, help="Regularization for contrastive loss")
    
    # Advanced training options
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--use_ema", action="store_true", help="Use exponential moving average model")
    
    # Distillation settings
    parser.add_argument("--train_baseline", action="store_true", help="Train baseline model for comparison")
    parser.add_argument("--distill", action="store_true", help="Use knowledge distillation")
    parser.add_argument("--distill_weight", type=float, default=0.3, help="Weight for distillation loss")
    
    # Memory optimization settings
    parser.add_argument("--grad_checkpointing", action="store_true", help="Use gradient checkpointing to save memory")
    
    args = parser.parse_args()
    train_retrieval(args)