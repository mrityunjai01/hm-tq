import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.nn2a.base_model import BaseHangmanModel


class HangmanNet(BaseHangmanModel):
    """
    Neural network for hangman letter prediction.

    Takes categorical input of shape (batch_size, 34, vocab_size) and
    outputs multilabel predictions of shape (batch_size, 26).
    """

    __constants__ = ["device"]

    def __init__(
        self,
        pos_embed_size=8,
        vocab_size=28,
        target_vocab_size=27,
        input_dim=34,
        hidden_dim1=256,
        hidden_dim2=64,
        dropout_rate=0.3,
        embed_dimension=None,
        device="cpu",
    ):
        super(HangmanNet, self).__init__()

        self.pos_embed_layer = nn.Embedding(pos_embed_size, pos_embed_size)
        self.layer1 = nn.Linear(vocab_size * input_dim + pos_embed_size, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.bn3 = nn.BatchNorm1d(hidden_dim2)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(hidden_dim2, target_vocab_size)  # Output 26 letters
        self._init_weights()

        self.device = device

    def forward(self, x, pos):
        # breakpoint()
        pos = self.pos_embed_layer(pos)

        flattened_x = x.view(x.size(0), -1)
        x = torch.cat(
            [
                flattened_x,
                pos,
            ],
            dim=1,
        )

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

        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.layer4(x)
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
