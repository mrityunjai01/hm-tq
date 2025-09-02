import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseHangmanModel


class HangmanNet(BaseHangmanModel):
    """
    Neural network for hangman letter prediction.

    Takes categorical input of shape (batch_size, 34, vocab_size) and
    outputs multilabel predictions of shape (batch_size, 26).
    """

    __constants__ = ["device"]

    def __init__(
        self,
        vocab_size=28,
        input_dim=34,
        hidden_dim1=256,
        hidden_dim2=64,
        dropout_rate=0.3,
        embed_dimension=None,
        device="cpu",
    ):
        super(HangmanNet, self).__init__()

        if embed_dimension is None:
            embed_dimension = vocab_size

        self.embed_layer = nn.Embedding(vocab_size, embed_dimension)
        self.layer1 = nn.Linear(embed_dimension, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1 * input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(hidden_dim1 * input_dim, 26)  # Output 26 letters
        self._init_weights()

        self.device = device

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 34, vocab_size)

        Returns:
            Output tensor of shape (batch_size, 26) with sigmoid activation
        """
        x = self.embed_layer(x)
        x = self.layer1(x)
        x = x.view(x.size(0), -1)  # Flatten for batch norm
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.layer3(x)
        return x

    def predict_numpy(self, x):
        with torch.no_grad():
            x = torch.tensor(x).to(self.device)
            return F.softmax(self(x), dim=1).cpu().detach().numpy()


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
