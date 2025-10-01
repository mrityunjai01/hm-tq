import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nn.nn2a.base_model import BaseHangmanModel


class HangmanNet(BaseHangmanModel):
    """
    BERT-style encoder for hangman masked language modeling.
    Takes masked word sequences and predicts original characters.
    """

    __constants__ = ["device", "hidden_dim"]

    def __init__(
        self,
        vocab_size=27,  # a-z + '{'
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.3,
        max_seq_len=20,
        device="cpu",
    ):
        super(HangmanNet, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            layer_norm_eps=5e-5,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection to character vocabulary
        self.output_projection = nn.Linear(hidden_dim, 26)  # Only predict a-z (not '{')

        # constants
        self.hidden_dim = hidden_dim
        self.device = device

        self._init_weights()

    def create_positional_encoding(self, seq_len):
        """Create sinusoidal positional encoding"""
        position = torch.arange(seq_len, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=self.device)
            * -(math.log(10000.0) / self.hidden_dim)
        )

        pos_encoding = torch.zeros(seq_len, self.hidden_dim, device=self.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.char_embedding(x)  # (batch_size, seq_len, hidden_dim)
        x += self.create_positional_encoding(seq_len).expand(batch_size, -1, -1)

        # Apply transformer encoder
        x = self.transformer(x)  # (batch_size, seq_len, hidden_dim)

        # Output projection for each position
        x = self.output_projection(x)  # (batch_size, seq_len, 26)

        return x

    def predict_numpy(self, x):
        # the caller should call torch.no_grad() and model.eval()
        x = x.to(self.device)
        if x.dim() == 1:  # Add batch dimension if needed
            x = x.unsqueeze(0)
        logits = self(x)
        return F.softmax(logits, dim=-1).cpu().detach().numpy()


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
