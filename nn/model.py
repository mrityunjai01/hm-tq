import torch
import torch.nn as nn
import torch.nn.functional as F


class HangmanNet(nn.Module):
    """
    Neural network for hangman letter prediction.
    
    Takes categorical input of shape (batch_size, 34, vocab_size) and
    outputs multilabel predictions of shape (batch_size, 26).
    """
    
    def __init__(self, vocab_size=27, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.3):
        """
        Initialize the network.
        
        Args:
            vocab_size: Size of vocabulary (27 for a-z + padding)
            hidden_dim1: Size of first hidden layer
            hidden_dim2: Size of second hidden layer
            dropout_rate: Dropout rate for regularization
        """
        super(HangmanNet, self).__init__()
        
        self.vocab_size = vocab_size
        input_size = 34 * vocab_size  # Flattened one-hot vectors
        
        # Three-layer network with batch normalization
        self.layer1 = nn.Linear(input_size, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.layer3 = nn.Linear(hidden_dim2, 26)  # Output 26 letters
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 34, vocab_size)
            
        Returns:
            Output tensor of shape (batch_size, 26) with sigmoid activation
        """
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch_size, 34 * vocab_size)
        
        # Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3 (output layer)
        x = self.layer3(x)
        x = torch.sigmoid(x)  # Sigmoid for multilabel classification
        
        return x


class MultiLabelLoss(nn.Module):
    """
    Multi-label binary cross-entropy loss with optional class weighting.
    """
    
    def __init__(self, pos_weight=None):
        """
        Initialize loss function.
        
        Args:
            pos_weight: Positive class weights for handling class imbalance
        """
        super(MultiLabelLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets):
        """
        Compute multi-label loss.
        
        Args:
            predictions: Predicted probabilities (batch_size, 26)
            targets: True binary labels (batch_size, 26)
            
        Returns:
            Scalar loss value
        """
        return F.binary_cross_entropy(predictions, targets, weight=self.pos_weight)


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
        dropout_rate=dropout_rate
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)