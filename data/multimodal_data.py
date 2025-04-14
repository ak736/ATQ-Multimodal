import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class MultimodalDataset(Dataset):
    """
    Dataset for handling multimodal data (image and text)
    
    This is a placeholder for future expansion to multimodal capabilities.
    For the current ATQ implementation, we're focusing on single modality.
    """
    def __init__(self, image_dir, text_file, transform=None):
        """
        Initialize multimodal dataset
        
        Args:
            image_dir: Directory containing images
            text_file: JSON file with text data and image mappings
            transform: Transforms to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load text data from JSON file
        with open(text_file, 'r') as f:
            self.data = json.load(f)
        
        # Create image paths
        self.image_paths = [os.path.join(image_dir, item['image_file']) 
                           for item in self.data]
        
        # Get text data
        self.texts = [item['text'] for item in self.data]
        
        # Get labels if available
        self.labels = [item.get('label', -1) for item in self.data]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get text
        text = self.texts[idx]
        
        # Get label if available
        label = self.labels[idx]
        
        return {
            'image': image,
            'text': text,
            'label': label if label != -1 else None
        }

def get_multimodal_data(batch_size=32, image_dir=None, text_file=None, subset_fraction=1.0):
    """
    Load and prepare multimodal dataset
    
    This is a placeholder for future multimodal expansion.
    
    Args:
        batch_size: Batch size for data loaders
        image_dir: Directory containing images
        text_file: JSON file with text data and image mappings
        subset_fraction: Fraction of dataset to use (0-1)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Check if files exist
    if image_dir is None or text_file is None or not os.path.exists(text_file):
        raise ValueError("Must provide valid image directory and text file")
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = MultimodalDataset(
        image_dir=image_dir,
        text_file=text_file,
        transform=train_transform
    )
    
    # Reduce dataset size if needed
    if subset_fraction < 1.0:
        subset_size = int(len(dataset) * subset_fraction)
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader