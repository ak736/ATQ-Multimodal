import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_data(batch_size=128, data_dir='./data', subset_fraction=0.2):
    """
    Load and prepare MNIST dataset
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store datasets
        subset_fraction: Fraction of dataset to use (0-1)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),  # Small rotations
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Reduce dataset size if needed
    if subset_fraction < 1.0:
        subset_size = int(len(train_dataset) * subset_fraction)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # Split into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Reduce test dataset size if needed
    if subset_fraction < 1.0:
        test_subset_size = int(len(test_dataset) * subset_fraction)
        test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
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

def get_fashion_mnist_data(batch_size=128, data_dir='./data', subset_fraction=0.2):
    """
    Load and prepare Fashion-MNIST dataset
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store datasets
        subset_fraction: Fraction of dataset to use (0-1)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),  # Small rotations
        transforms.RandomHorizontalFlip(),  # Flip horizontally
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Download and load training data
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Reduce dataset size if needed
    if subset_fraction < 1.0:
        subset_size = int(len(train_dataset) * subset_fraction)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # Split into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Download and load test data
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Reduce test dataset size if needed
    if subset_fraction < 1.0:
        test_subset_size = int(len(test_dataset) * subset_fraction)
        test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
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