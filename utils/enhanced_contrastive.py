import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss


class HardNegativeMiningInfoNCE(_Loss):
    """
    InfoNCE loss with hard negative mining for better contrastive learning
    
    This enhances the basic InfoNCE loss by:
    1. Mining hard negatives in the batch
    2. Using temperature scaling with annealing
    3. Adding a regularization term to prevent feature collapse
    4. Supporting weighted positives for better training
    """
    def __init__(self, temperature=0.07, lambda_reg=0.02, hard_negative_weight=0.5, 
                 hardest_mining_ratio=0.5, temperature_schedule=True):
        """
        Initialize InfoNCE loss with hard negative mining
        
        Args:
            temperature: Base temperature parameter for similarity scaling
            lambda_reg: Regularization coefficient to prevent feature collapse
            hard_negative_weight: Weight for hard negative samples (0.0 to 1.0)
            hardest_mining_ratio: Ratio of hardest negatives to use per sample (0.0 to 1.0)
            temperature_schedule: Whether to use temperature annealing during training
        """
        super(HardNegativeMiningInfoNCE, self).__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.hard_negative_weight = hard_negative_weight
        self.hardest_mining_ratio = hardest_mining_ratio
        self.temperature_schedule = temperature_schedule
        self.base_temperature = temperature
        
        # For temperature scheduling
        self.current_epoch = 0
        self.total_epochs = 1
    
    def set_epoch(self, current_epoch, total_epochs):
        """Set current epoch for temperature scheduling"""
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
    
    def get_current_temperature(self):
        """Get temperature based on training schedule"""
        if not self.temperature_schedule:
            return self.temperature
        
        # Start with higher temperature (softer distribution)
        # and anneal to lower temperature (sharper distribution)
        progress = min(1.0, self.current_epoch / (self.total_epochs * 0.7))
        max_temp = self.base_temperature * 2.0
        min_temp = self.base_temperature * 0.5
        
        # Cosine annealing schedule
        temperature = max_temp - (max_temp - min_temp) * (1 - np.cos(progress * np.pi)) / 2
        
        # Keep temperature in safe range
        return max(min(temperature, max_temp), min_temp)
    
    def forward(self, image_embeddings, text_embeddings, weights=None):
        """
        Forward pass for InfoNCE loss with hard negative mining
        
        Args:
            image_embeddings: Image embeddings [batch_size, embed_dim]
            text_embeddings: Text embeddings [batch_size, embed_dim]
            weights: Optional weights for positive pairs [batch_size]
            
        Returns:
            Loss value
        """
        # Get current temperature
        temperature = self.get_current_temperature()
        
        # Normalize embeddings for cosine similarity
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        # sim[i, j] = similarity between image i and text j
        similarity = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
        
        batch_size = image_embeddings.size(0)
        
        # Create labels for matching pairs (diagonal elements)
        labels = torch.arange(batch_size, device=similarity.device)
        
        # Create masks for positive and negative pairs
        pos_mask = torch.zeros_like(similarity)
        pos_mask.fill_diagonal_(1)
        neg_mask = 1 - pos_mask
        
        # Find hard negatives (highest similarity among negatives)
        with torch.no_grad():
            # For each image, find hardest text negatives
            # Set diagonal to -inf to exclude positives
            sim_img_to_txt = similarity.clone()
            sim_img_to_txt.fill_diagonal_(-float('inf'))
            
            # Get top-k hardest negatives
            k = max(1, int(batch_size * self.hardest_mining_ratio))
            hard_img_to_txt_values, hard_img_to_txt_indices = sim_img_to_txt.topk(k, dim=1)
            
            # For each text, find hardest image negatives
            sim_txt_to_img = similarity.t().clone()
            sim_txt_to_img.fill_diagonal_(-float('inf'))
            hard_txt_to_img_values, hard_txt_to_img_indices = sim_txt_to_img.topk(k, dim=1)
            
            # Create hard negative masks
            hard_neg_mask_img = torch.zeros_like(similarity)
            hard_neg_mask_txt = torch.zeros_like(similarity)
            
            # Fill in hard negative masks
            for i in range(batch_size):
                hard_neg_mask_img[i, hard_img_to_txt_indices[i]] = 1
                hard_neg_mask_txt[hard_txt_to_img_indices[i], i] = 1
            
            # Combine hard negative masks
            hard_neg_mask = (hard_neg_mask_img + hard_neg_mask_txt) > 0
            hard_neg_mask = hard_neg_mask.float() * neg_mask
            
            # Final negative mask: regular negatives and hard negatives
            easy_neg_mask = neg_mask - hard_neg_mask
        
        # Apply weights for positives if provided
        pos_weights = weights if weights is not None else torch.ones(batch_size, device=similarity.device)
        pos_weights = pos_weights.view(-1, 1)  # [batch_size, 1]
        
        # Weight hard negatives higher than easy negatives
        neg_weights = torch.ones_like(similarity)
        neg_weights = neg_weights * easy_neg_mask + \
                      neg_weights * hard_neg_mask * (1.0 + self.hard_negative_weight)
        
        # Create weighted similarity scores
        weighted_similarity = similarity * pos_mask * pos_weights + \
                              similarity * neg_weights
        
        # Compute InfoNCE loss in both directions
        image_loss = self._cross_entropy_loss(weighted_similarity, labels)
        text_loss = self._cross_entropy_loss(weighted_similarity.t(), labels)
        
        # Regularization to prevent feature collapse
        img_entropy = -torch.mean(torch.sum(F.softmax(similarity, dim=1) * 
                                           F.log_softmax(similarity, dim=1), dim=1))
        txt_entropy = -torch.mean(torch.sum(F.softmax(similarity.t(), dim=1) * 
                                           F.log_softmax(similarity.t(), dim=1), dim=1))
        
        # Higher entropy = more uniform distribution = less collapse
        regularity_loss = self.lambda_reg * (img_entropy + txt_entropy) / 2
        
        # Total loss
        total_loss = (image_loss + text_loss) / 2 + regularity_loss
        
        return total_loss
    
    def _cross_entropy_loss(self, similarity, labels):
        """Compute cross entropy loss"""
        return F.cross_entropy(similarity, labels)


class MultiPositiveInfoNCE(_Loss):
    """
    InfoNCE loss with multiple positives support
    
    This handles cases where each anchor has multiple positive examples,
    such as an image with multiple captions.
    """
    def __init__(self, temperature=0.07, lambda_reg=0.02):
        """
        Initialize InfoNCE loss with multiple positives
        
        Args:
            temperature: Temperature parameter for similarity scaling
            lambda_reg: Regularization coefficient to prevent feature collapse
        """
        super(MultiPositiveInfoNCE, self).__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
    
    def forward(self, image_embeddings, text_embeddings, positive_mask):
        """
        Forward pass for InfoNCE loss with multiple positives
        
        Args:
            image_embeddings: Image embeddings [batch_size, embed_dim]
            text_embeddings: Text embeddings [batch_size, embed_dim]
            positive_mask: Binary mask indicating positive pairs [batch_size, batch_size]
                          positive_mask[i, j] = 1 if image i and text j are positive pairs
            
        Returns:
            Loss value
        """
        # Normalize embeddings for cosine similarity
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        # sim[i, j] = similarity between image i and text j
        similarity = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        batch_size = image_embeddings.size(0)
        
        # Create mask for negative pairs
        neg_mask = 1 - positive_mask
        
        # Image-to-text loss (for each image, match with positive texts)
        i2t_loss = 0.0
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i] == 1)[0]
            if len(pos_indices) == 0:
                continue  # Skip if no positives
                
            # Image i has positive texts at pos_indices
            # Create a new similarity row with just the positives having high values
            i_sim = similarity[i]
            
            # Create a label distribution that equally weights all positives
            target_dist = torch.zeros_like(i_sim)
            target_dist[pos_indices] = 1.0 / len(pos_indices)
            
            # Cross entropy between similarity and target distribution
            i_loss = -torch.sum(target_dist * F.log_softmax(i_sim, dim=0))
            i2t_loss += i_loss
        
        # Normalize by number of images with positives
        i2t_loss = i2t_loss / batch_size
        
        # Text-to-image loss (for each text, match with positive images)
        t2i_loss = 0.0
        for j in range(batch_size):
            pos_indices = torch.where(positive_mask[:, j] == 1)[0]
            if len(pos_indices) == 0:
                continue  # Skip if no positives
                
            # Text j has positive images at pos_indices
            # Create a new similarity column with just the positives having high values
            j_sim = similarity[:, j]
            
            # Create a label distribution that equally weights all positives
            target_dist = torch.zeros_like(j_sim)
            target_dist[pos_indices] = 1.0 / len(pos_indices)
            
            # Cross entropy between similarity and target distribution
            j_loss = -torch.sum(target_dist * F.log_softmax(j_sim, dim=0))
            t2i_loss += j_loss
        
        # Normalize by number of texts with positives
        t2i_loss = t2i_loss / batch_size
        
        # Regularization to prevent feature collapse
        img_entropy = -torch.mean(torch.sum(F.softmax(similarity, dim=1) * 
                                           F.log_softmax(similarity, dim=1), dim=1))
        txt_entropy = -torch.mean(torch.sum(F.softmax(similarity.t(), dim=1) * 
                                           F.log_softmax(similarity.t(), dim=1), dim=1))
        
        # Higher entropy = more uniform distribution = less collapse
        regularity_loss = -self.lambda_reg * (img_entropy + txt_entropy) / 2
        
        # Total loss
        total_loss = (i2t_loss + t2i_loss) / 2 + regularity_loss
        
        return total_loss


class ContrastiveLearningManager:
    """
    Manager for contrastive learning with curriculum and hard example mining
    """
    def __init__(self, model, criterion, similarity_threshold=0.8, 
                 mining_freq=50, curriculum_steps=3):
        """
        Initialize contrastive learning manager
        
        Args:
            model: Model for inference
            criterion: Contrastive loss function
            similarity_threshold: Threshold to consider examples as similar
            mining_freq: Frequency of mining operations (batches)
            curriculum_steps: Number of curriculum difficulty steps
        """
        self.model = model
        self.criterion = criterion
        self.similarity_threshold = similarity_threshold
        self.mining_freq = mining_freq
        self.curriculum_steps = curriculum_steps
        
        # Mining state
        self.steps = 0
        self.mined_examples = []
        self.epoch = 0
        self.total_epochs = 0
        
        # Curriculum learning state
        self.curriculum_stage = 0
    
    def set_epoch(self, epoch, total_epochs):
        """Update epoch information for curriculum learning"""
        self.epoch = epoch
        self.total_epochs = total_epochs
        
        # Update curriculum stage based on training progress
        progress = epoch / total_epochs
        self.curriculum_stage = min(self.curriculum_steps - 1, 
                                   int(progress * self.curriculum_steps))
    
    def get_curriculum_weight(self, similarity):
        """
        Get curriculum-adjusted weights based on pair difficulty
        
        Args:
            similarity: Similarity matrix [batch_size, batch_size]
            
        Returns:
            Weights for positive pairs [batch_size]
        """
        batch_size = similarity.size(0)
        
        # Extract positive pair similarities (diagonal)
        pos_similarities = torch.diag(similarity)
        
        # Calculate weights based on similarity and curriculum stage
        # Early stages: focus on easy positives (high similarity)
        # Later stages: focus on hard positives (low similarity)
        if self.curriculum_stage == 0:  # Early stage - easy examples
            # Higher weight for high similarity (easy) pairs
            weights = torch.sigmoid(pos_similarities * 10)
        elif self.curriculum_stage == self.curriculum_steps - 1:  # Final stage - hard examples
            # Higher weight for low similarity (hard) pairs
            weights = 1 - torch.sigmoid(pos_similarities * 10 - 5)
        else:  # Middle stages - balanced
            # Uniform weights
            weights = torch.ones_like(pos_similarities)
        
        return weights
    
    def mine_hard_examples(self, dataloader, device, max_examples=1000):
        """
        Mine hard positive and negative examples from the dataset
        
        Args:
            dataloader: DataLoader to mine examples from
            device: Device to use for computation
            max_examples: Maximum number of examples to mine
            
        Returns:
            List of mined example indices
        """
        self.model.eval()
        hard_examples = []
        
        with torch.no_grad():
            for batch_idx, (images, captions, lengths) in enumerate(dataloader):
                if len(hard_examples) >= max_examples:
                    break
                    
                # Move data to device
                images = images.to(device)
                captions = captions.to(device)
                if not torch.is_tensor(lengths):
                    lengths = torch.tensor(lengths, device=device)
                elif lengths.device != device:
                    lengths = lengths.to(device)
                
                # Get embeddings
                image_embeddings, text_embeddings = self.model(
                    images, captions, lengths, return_embeddings=True
                )
                
                # Compute similarity matrix
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
                similarity = torch.matmul(image_embeddings, text_embeddings.t())
                
                # Find hard positives (low similarity for matching pairs)
                pos_similarities = torch.diag(similarity)
                hard_positive_indices = torch.where(pos_similarities < self.similarity_threshold)[0]
                
                # Add hard positives to list
                for idx in hard_positive_indices:
                    if len(hard_examples) < max_examples:
                        hard_examples.append(batch_idx * len(images) + idx.item())
        
        self.mined_examples = hard_examples
        return hard_examples
    
    def compute_loss(self, image_embeddings, text_embeddings, similarity=None):
        """
        Compute contrastive loss with curriculum and mining
        
        Args:
            image_embeddings: Image embeddings [batch_size, embed_dim]
            text_embeddings: Text embeddings [batch_size, embed_dim]
            similarity: Optional pre-computed similarity matrix
            
        Returns:
            Loss value
        """
        # Increment step counter
        self.steps += 1
        
        # Compute similarity if not provided
        if similarity is None:
            image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=1)
            similarity = torch.matmul(image_embeddings_norm, text_embeddings_norm.t())
        
        # Get curriculum weights
        weights = self.get_curriculum_weight(similarity)
        
        # Compute loss with weights
        loss = self.criterion(image_embeddings, text_embeddings, weights)
        
        return loss
        
        
# Example usage
def train_with_enhanced_contrastive(model, train_loader, optimizer, device, epoch, total_epochs):
    """
    Training loop with enhanced contrastive learning
    
    Args:
        model: Model to train
        train_loader: DataLoader for training
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch
        total_epochs: Total epochs
    """
    # Create contrastive loss with hard negative mining
    criterion = HardNegativeMiningInfoNCE(
        temperature=0.07,
        lambda_reg=0.02,
        hard_negative_weight=0.5,
        temperature_schedule=True
    )
    criterion.set_epoch(epoch, total_epochs)
    
    # Create contrastive learning manager
    cl_manager = ContrastiveLearningManager(
        model=model,
        criterion=criterion,
        similarity_threshold=0.7,
        mining_freq=100
    )
    cl_manager.set_epoch(epoch, total_epochs)
    
    # Set model to training mode
    model.train()
    
    for batch_idx, (images, captions, lengths) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        captions = captions.to(device)
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=device)
        elif lengths.device != device:
            lengths = lengths.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass - get similarity matrix
        image_embeddings, text_embeddings = model(
            images, captions, lengths, return_embeddings=True
        )
        
        # Compute loss with curriculum and mining
        loss = cl_manager.compute_loss(image_embeddings, text_embeddings)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return loss.item()