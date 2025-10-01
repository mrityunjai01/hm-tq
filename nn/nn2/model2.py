import torch
import torch.nn as nn
import torch.nn.functional as F


class HangmanNet(nn.Module):
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

        self.layer2 = nn.Linear(hidden_dim1 * input_dim, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.bn3 = nn.BatchNorm1d(hidden_dim2)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.bn4 = nn.BatchNorm1d(hidden_dim2)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.layer5 = nn.Linear(hidden_dim2, 26)  # Output 26 letters
        self._init_weights()

        self.device = device

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_last_embedding(self):
        """ensure the last embedding is always the same"""
        mask = torch.ones(
            (self.embed_layer.weight.shape[0], self.embed_layer.weight.shape[0])
        )
        mask[:, -1] = 0
        mask /= mask.sum(dim=1, keepdim=True)
        self.embed_layer.weight.data = torch.matmul(mask, self.embed_layer.weight)

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

        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        rec = self.layer3(x)
        rec = F.relu(rec)
        rec = self.bn3(rec)
        rec = self.dropout3(rec)

        x = x + rec

        rec = self.layer4(x)
        rec = self.bn4(rec)
        rec = F.relu(rec)
        rec = self.dropout4(rec)

        x = x + rec
        x = self.layer5(x)
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
