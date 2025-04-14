import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json

class MultimodalDataset(Dataset):
    """
    Dataset for multimodal data (image and text)
    """
    def __init__(self, image_dir, text_file, vocab=None, max_length=100, image_transform=None):
        """
        Initialize multimodal dataset
        
        Args:
            image_dir: Directory containing images
            text_file: JSON file mapping image files to text and labels
            vocab: Vocabulary for text tokenization (if None, will return raw text)
            max_length: Maximum sequence length for text
            image_transform: Transforms to apply to images
        """
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_length = max_length
        self.image_transform = image_transform
        
        # Load data
        with open(text_file, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # Process text
        text = item['text']
        
        if self.vocab:
            # Tokenize text using vocabulary
            tokens = text.split()
            token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
            
            # Pad or truncate to max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_length - len(token_ids))
            
            text_tensor = torch.tensor(token_ids, dtype=torch.long)
            text_length = min(len(tokens), self.max_length)
        else:
            # Return raw text if no vocabulary is provided
            text_tensor = text
            text_length = len(text.split())
        
        # Get label
        label = item.get('label', -1)
        if label != -1:
            label = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': image,
            'text': text_tensor,
            'text_length': text_length,
            'label': label if label != -1 else None
        }