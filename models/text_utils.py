import torch
import re
import nltk
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def build_vocabulary(text_data, min_freq=2, max_size=10000):
    """
    Build vocabulary from text data
    
    Args:
        text_data: List of text strings
        min_freq: Minimum frequency for a token to be included
        max_size: Maximum vocabulary size
        
    Returns:
        Dictionary mapping tokens to indices
    """
    # Tokenize all texts
    all_tokens = []
    for text in text_data:
        tokens = word_tokenize(text.lower())
        all_tokens.extend(tokens)
    
    # Count token frequencies
    counter = Counter(all_tokens)
    
    # Filter by frequency and limit size
    vocab_tokens = [token for token, count in counter.most_common(max_size) if count >= min_freq]
    
    # Add special tokens
    vocab = {
        '<PAD>': 0,  # Padding token
        '<UNK>': 1,  # Unknown token
        '<SOS>': 2,  # Start of sequence
        '<EOS>': 3   # End of sequence
    }
    
    # Add regular tokens
    for i, token in enumerate(vocab_tokens):
        vocab[token] = i + 4  # Start after special tokens
    
    return vocab

def clean_text(text):
    """
    Clean text by removing punctuation, numbers, and stopwords
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def tokenize_texts(texts, vocab, max_length=100):
    """
    Tokenize multiple texts using provided vocabulary
    
    Args:
        texts: List of text strings
        vocab: Vocabulary dictionary
        max_length: Maximum sequence length
        
    Returns:
        Tensor of token IDs and list of sequence lengths
    """
    batch_size = len(texts)
    token_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    lengths = []
    
    for i, text in enumerate(texts):
        tokens = word_tokenize(text.lower())
        length = min(len(tokens), max_length)
        lengths.append(length)
        
        # Convert tokens to IDs
        for j, token in enumerate(tokens[:max_length]):
            token_ids[i, j] = vocab.get(token, vocab['<UNK>'])
    
    return token_ids, lengths

def create_embedding_matrix(vocab, embedding_dim=300, pretrained_file=None):
    """
    Create embedding matrix from vocabulary
    
    Args:
        vocab: Vocabulary dictionary
        embedding_dim: Dimension of embeddings
        pretrained_file: Optional pretrained embeddings file
        
    Returns:
        Embedding matrix as tensor
    """
    vocab_size = len(vocab)
    embedding_matrix = torch.randn(vocab_size, embedding_dim) * 0.1
    
    # Set padding token to all zeros
    embedding_matrix[vocab['<PAD>']] = torch.zeros(embedding_dim)
    
    # Load pretrained embeddings if provided
    if pretrained_file:
        pretrained = {}
        with open(pretrained_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = torch.tensor([float(x) for x in values[1:]])
                pretrained[word] = vector
        
        # Update embedding matrix with pretrained vectors
        for word, idx in vocab.items():
            if word in pretrained:
                embedding_matrix[idx] = pretrained[word]
    
    return embedding_matrix