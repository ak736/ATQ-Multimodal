import torch
import torch.nn as nn
import torch.nn.functional as F
from atq.layers import TernaryLinear
from atq.precision_boost import ResidualPrecisionBoostLinear
from atq.routing import apply_selective_routing

class ATQTextEncoder(nn.Module):
    """
    Text encoder using Adaptive Ternary Quantization
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, num_layers=2, use_rpb=True):
        super(ATQTextEncoder, self).__init__()
        self.use_rpb = use_rpb
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer (not quantized)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for text encoding (not quantized as it's not fully custom)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output dimension after bidirectional LSTM
        lstm_output_dim = hidden_dim * 2
        
        # Quantized path
        if use_rpb:
            self.fc_quant = nn.Sequential(
                ResidualPrecisionBoostLinear(lstm_output_dim, hidden_dim, precision_ratio=0.1),
                nn.ReLU(),
                ResidualPrecisionBoostLinear(hidden_dim, hidden_dim, precision_ratio=0.1)
            )
        else:
            self.fc_quant = nn.Sequential(
                TernaryLinear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                TernaryLinear(hidden_dim, hidden_dim)
            )
        
        # Full-precision path
        self.fc_full = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Mixing parameter
        self.mix_ratio = nn.Parameter(torch.tensor(0.75))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, lengths=None):
        # x shape: (batch_size, sequence_length)
        
        # Embed text
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence if lengths are provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, _) = self.lstm(packed)
            
            # Unpack the sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, _) = self.lstm(embedded)
        
        # Use hidden state from last layer as the text representation
        # Concatenate forward and backward hidden states
        hidden_forward = hidden[2*self.num_layers-2, :, :]
        hidden_backward = hidden[2*self.num_layers-1, :, :]
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Apply selective gradient routing
        hidden_route = apply_selective_routing(hidden_cat, threshold=0.05)
        
        # Get outputs from both paths
        out_quant = self.fc_quant(hidden_route)
        out_full = self.fc_full(hidden_cat)
        
        # Mix outputs
        mix = torch.sigmoid(self.mix_ratio)
        out = mix * out_quant + (1 - mix) * out_full
        
        return out