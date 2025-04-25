import os
import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import re
import requests
import zipfile
import io
from tqdm import tqdm
import ssl
import nltk
from collections import Counter

def setup_nltk():
    """Set up NLTK with proper data downloads and paths"""
    import os
    
    # Handle SSL certificate issue
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Set NLTK data path to a more standard location
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    
    # Create the directory if it doesn't exist
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add this to NLTK's search path
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    # Try to download punkt and punkt_tab
    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    
    # Try to load the tokenizer to verify it's working
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK punkt tokenizer successfully loaded")
        return True
    except LookupError:
        print("NLTK punkt tokenizer still not available")
        return False

class Flickr8kDataset(Dataset):
    """
    Dataset for Flickr8k dataset for image-text retrieval
    """
    def __init__(self, root_dir='./data/flickr8k', transform=None, split='train', 
                max_length=50, download=True, tokenize_captions=True):
        """
        Initialize Flickr8k dataset
        
        Args:
            root_dir: Root directory for the dataset
            transform: Transforms to apply to the images
            split: 'train', 'val', or 'test' split
            max_length: Maximum caption length
            download: Whether to download the dataset if not found
            tokenize_captions: Whether to tokenize captions (if False, returns raw text)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.max_length = max_length
        self.tokenize_captions = tokenize_captions
        
        # Set up NLTK
        self.use_nltk = setup_nltk()
        
        # Download dataset if needed
        if download and not self._check_exists():
            self._download_and_extract()
        
        # Load dataset
        self._load_dataset()
        
        # Build vocabulary if tokenizing
        if tokenize_captions:
            self._build_vocabulary()
    
    def _check_exists(self):
        """Check if the dataset exists"""
        image_dir = os.path.join(self.root_dir, 'Flicker8k_Dataset')
        caption_file = os.path.join(self.root_dir, 'Flickr8k.token.txt')
        return os.path.exists(image_dir) and os.path.exists(caption_file)
    
    def _download_and_extract(self):
        """Download and extract the Flickr8k dataset"""
        print("Downloading Flickr8k dataset...")
        
        # Create directory
        os.makedirs(self.root_dir, exist_ok=True)
        
        # URLs for the dataset
        # Note: These URLs may change. If they don't work, you'll need to manually download.
        image_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
        text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
        
        # Download and extract images
        try:
            print("Downloading images...")
            response = requests.get(image_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(self.root_dir)
            
            # Download and extract text
            print("Downloading captions...")
            response = requests.get(text_url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(self.root_dir)
                
            print("Dataset downloaded and extracted successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download the dataset manually:")
            print("1. Download Flickr8k_Dataset.zip and Flickr8k_text.zip")
            print("2. Extract them to the data/flickr8k directory")
            raise RuntimeError("Dataset download failed")
    
    def _load_dataset(self):
        """Load the dataset"""
        # Load captions
        caption_file = os.path.join(self.root_dir, 'Flickr8k.token.txt')
        
        # Load captions into a dictionary: {image_name: [caption1, caption2, ...]}
        self.captions = {}
        with open(caption_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: {image_name}#{caption_number}\t{caption}
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                image_caption_id, caption = parts
                image_name = image_caption_id.split('#')[0]
                
                if image_name not in self.captions:
                    self.captions[image_name] = []
                
                # Clean caption (remove special chars and lowercase)
                caption = caption.lower()
                caption = re.sub(r'[^\w\s]', '', caption)
                self.captions[image_name].append(caption)
        
        # Load train/val/test split
        train_file = os.path.join(self.root_dir, 'Flickr_8k.trainImages.txt')
        val_file = os.path.join(self.root_dir, 'Flickr_8k.devImages.txt')
        test_file = os.path.join(self.root_dir, 'Flickr_8k.testImages.txt')
        
        # If split files don't exist, create our own split (80/10/10)
        if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
            all_images = list(self.captions.keys())
            np.random.shuffle(all_images)
            
            train_size = int(0.8 * len(all_images))
            val_size = int(0.1 * len(all_images))
            
            train_images = all_images[:train_size]
            val_images = all_images[train_size:train_size+val_size]
            test_images = all_images[train_size+val_size:]
            
            # Save splits
            os.makedirs(os.path.join(self.root_dir, 'Flickr8k_text'), exist_ok=True)
            
            with open(train_file, 'w') as f:
                f.write('\n'.join(train_images))
            
            with open(val_file, 'w') as f:
                f.write('\n'.join(val_images))
            
            with open(test_file, 'w') as f:
                f.write('\n'.join(test_images))
        else:
            # Load existing splits
            train_images = []
            with open(train_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        train_images.append(line)
            
            val_images = []
            with open(val_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        val_images.append(line)
            
            test_images = []
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        test_images.append(line)
        
        # Select the appropriate split
        if self.split == 'train':
            self.image_names = train_images
        elif self.split == 'val':
            self.image_names = val_images
        elif self.split == 'test':
            self.image_names = test_images
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Create dataset items: (image_name, caption)
        self.items = []
        for image_name in self.image_names:
            if image_name in self.captions:
                for caption in self.captions[image_name]:
                    self.items.append((image_name, caption))
        
        print(f"Loaded {len(self.items)} image-caption pairs for {self.split} split")
    
    def _build_vocabulary(self):
        """Build vocabulary from captions using NLTK or simple tokenization"""
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        idx = 4
        
        # Count word frequencies
        word_counts = {}
        
        for _, caption in self.items:
            if self.use_nltk:
                try:
                    words = nltk.tokenize.word_tokenize(caption.lower())
                except Exception as e:
                    print(f"NLTK tokenization failed, using simple tokenization: {e}")
                    words = caption.lower().split()
            else:
                words = caption.lower().split()
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create vocabulary (only include words that appear at least 5 times)
        for word, count in word_counts.items():
            if count >= 5:  # Frequency threshold
                self.word_to_idx[word] = idx
                idx += 1
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image_name, caption = self.items[idx]
        
        # Load and process image
        image_path = os.path.join(self.root_dir, 'Flicker8k_Dataset', image_name)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process caption
        if self.tokenize_captions:
            # Tokenize and convert to indices
            if self.use_nltk:
                try:
                    tokens = nltk.tokenize.word_tokenize(caption.lower())
                except:
                    tokens = caption.lower().split()
            else:
                tokens = caption.lower().split()
            
            token_ids = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
            
            # Add start and end tokens
            token_ids = [self.word_to_idx['<START>']] + token_ids + [self.word_to_idx['<END>']]
            
            # Pad or truncate
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids = token_ids + [self.word_to_idx['<PAD>']] * (self.max_length - len(token_ids))
            
            caption_tensor = torch.tensor(token_ids, dtype=torch.long)
            caption_length = min(len(tokens) + 2, self.max_length)  # +2 for <START> and <END>
            
            return image, caption_tensor, caption_length
        else:
            # Return raw caption
            return image, caption, len(caption.split())


def prepare_flickr8k_dataloaders(batch_size=32, image_size=224, max_length=50, tokenize_captions=True, num_workers=2):
    """
    Prepare Flickr8k dataloaders for training, validation and testing
    
    Args:
        batch_size: Batch size for dataloaders
        image_size: Size to resize images to
        max_length: Maximum caption length
        tokenize_captions: Whether to tokenize captions
        num_workers: Number of workers for dataloaders
        
    Returns:
        train_loader, val_loader, test_loader, vocabulary size, vocabulary
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = Flickr8kDataset(
        split='train',
        transform=transform,
        max_length=max_length,
        tokenize_captions=tokenize_captions
    )
    
    val_dataset = Flickr8kDataset(
        split='val',
        transform=test_transform,
        max_length=max_length,
        tokenize_captions=tokenize_captions
    )
    
    test_dataset = Flickr8kDataset(
        split='test',
        transform=test_transform,
        max_length=max_length,
        tokenize_captions=tokenize_captions
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Return vocabulary information if tokenizing
    if tokenize_captions:
        return train_loader, val_loader, test_loader, train_dataset.vocab_size, train_dataset.word_to_idx
    else:
        return train_loader, val_loader, test_loader, None, None


def visualize_flickr8k_samples(dataloader, num_samples=5, idx_to_word=None):
    """
    Visualize samples from the Flickr8k dataset
    
    Args:
        dataloader: DataLoader for Flickr8k dataset
        num_samples: Number of samples to visualize
        idx_to_word: Index to word mapping (for tokenized captions)
    """
    # Get a batch
    data = next(iter(dataloader))
    
    if len(data) == 3:
        images, captions, lengths = data
        tokenized = True
    else:
        images, captions = data
        tokenized = False
    
    # Visualize samples
    plt.figure(figsize=(15, 5*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Get image
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Get caption
        if tokenized and idx_to_word:
            cap_ids = captions[i].tolist()
            # Convert indices to words (skip padding)
            cap_words = []
            for idx in cap_ids:
                if idx == 0:  # <PAD>
                    continue
                if idx == 3:  # <END>
                    break
                if idx >= 2:  # Skip special tokens
                    if idx in idx_to_word:
                        cap_words.append(idx_to_word[idx])
                    else:
                        cap_words.append('<UNK>')
            
            caption = ' '.join(cap_words)
        else:
            caption = captions[i] if not torch.is_tensor(captions[i]) else "No caption available"
        
        # Display image and caption
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img)
        plt.title(f"Caption: {caption}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('flickr8k_samples.png')
    plt.close()
    
    print(f"Visualization saved to 'flickr8k_samples.png'")


# Example of usage
if __name__ == "__main__":
    # Prepare dataloaders
    train_loader, val_loader, test_loader, vocab_size, word_to_idx = prepare_flickr8k_dataloaders(
        batch_size=8,
        image_size=224,
        max_length=50
    )
    
    # Create idx_to_word mapping
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Visualize samples
    visualize_flickr8k_samples(train_loader, num_samples=5, idx_to_word=idx_to_word)
    
    # Print dataset information
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Vocabulary size: {vocab_size}")