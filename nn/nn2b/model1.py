import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nn.nn2a.base_model import BaseHangmanModel


class HangmanNet(BaseHangmanModel):
    """
    Neural network for hangman letter prediction with multi-head attention.

    Takes word sequences and uses self-attention with diagonal masking.
    """

    __constants__ = ["device"]

    def __init__(
        self,
        pos_embed_size=8,
        vocab_size=28,
        target_vocab_size=26,
        seq_len=10,  # Maximum word length
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dropout_rate=0.3,
        device="cpu",
    ):
        super(HangmanNet, self).__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Position embedding
        self.pos_embed_layer = nn.Embedding(pos_embed_size, hidden_dim)
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(vocab_size, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feedforward layers after each attention
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_layers)
        ])
        
        self.feedforward_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final projection to target vocabulary
        self.output_projection = nn.Linear(hidden_dim, target_vocab_size)
        
        self._init_weights()

    def create_eye_mask(self, seq_len):
        """Create attention mask to block diagonal (self-attention)"""
        mask = torch.eye(seq_len, device=self.device, dtype=torch.bool)
        # Convert to additive mask: True positions are masked (set to -inf)
        return mask

    def forward(self, x, pos):
        batch_size, seq_len, vocab_size = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add position embeddings
        pos_embeds = self.pos_embed_layer(pos)  # (batch_size, hidden_dim)
        pos_embeds = pos_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        x = x + pos_embeds
        
        # Create eye mask to prevent self-attention
        attn_mask = self.create_eye_mask(seq_len)
        
        # Apply attention layers
        for i, (attention, layer_norm, feedforward, ff_norm) in enumerate(
            zip(self.attention_layers, self.layer_norms, self.feedforward_layers, self.feedforward_norms)
        ):
            # Multi-head attention with residual connection
            residual = x
            x, _ = attention(x, x, x, attn_mask=attn_mask)
            x = layer_norm(x + residual)
            
            # Feedforward with residual connection
            residual = x
            x = feedforward(x)
            x = ff_norm(x + residual)
        
        # Global pooling across sequence dimension
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)
        
        # Final projection
        x = self.output_projection(x)  # (batch_size, target_vocab_size)
        return x

    def predict_numpy(self, x, pos):
        # the caller should call torch.no_grad() and model.eval() because of batchnorm and autograd
        x = torch.tensor(x).to(self.device)
        pos = torch.tensor(pos).to(self.device)
        return F.softmax(self(x, pos), dim=1).cpu().detach().numpy()


def create_model(vocab_size=27, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.3):
    """
    Create and return a HangmanNet model.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim1: Size of first hidden layer
        hidden_dim2: Size of second hidden layer
        dropout_rate: Dropout rate

    Returns:
        HangmanNet model
    """
    model = HangmanNet(
        vocab_size=vocab_size,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout_rate=dropout_rate,
    )

    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
