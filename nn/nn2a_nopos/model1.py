import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.nn2a_nopos.base_model import BaseHangmanModel


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

        self.layer1 = nn.Linear(vocab_size * input_dim, hidden_dim1)
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

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten for batch norm
        x = self.layer1(x)
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

    def predict_numpy(
        self,
        x,
    ):
        # the caller should call torch.no_grad() and model.eval() because of batchnorm and autograd
        x = torch.tensor(x).to(self.device)
        return F.softmax(self(x), dim=1).cpu().detach().numpy()


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
