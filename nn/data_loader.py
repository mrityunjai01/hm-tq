import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from prepare.create_dataset import load_dataset


class HangmanDataset(Dataset):
    """Dataset class for hangman neural network training."""
    
    def __init__(self, X, Y, vocab_size=27):  # 26 letters + 1 for padding/unknown
        """
        Initialize dataset with preprocessing.
        
        Args:
            X: Input features (N, 34) with values -1 to 26
            Y: Target labels (N, 26) binary multilabel
            vocab_size: Size of vocabulary (27 for a-z + padding)
        """
        self.vocab_size = vocab_size
        self.X = self._preprocess_features(X)
        self.Y = torch.FloatTensor(Y.astype(np.float32))
        
    def _preprocess_features(self, X):
        """
        Convert categorical features to one-hot encoding.
        
        Args:
            X: Raw features with values -1 to 26
            
        Returns:
            One-hot encoded features (N, 34, vocab_size)
        """
        # Shift values to 0-26 range (0 for unknown/blank, 1-26 for a-z)
        X_shifted = X + 1  # -1 becomes 0, 0 becomes 1, ..., 26 becomes 27
        X_shifted = np.clip(X_shifted, 0, self.vocab_size - 1)
        
        # One-hot encode each position
        batch_size, seq_len = X_shifted.shape
        X_onehot = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
        
        for i in range(batch_size):
            for j in range(seq_len):
                X_onehot[i, j, X_shifted[i, j]] = 1.0
                
        return torch.FloatTensor(X_onehot)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_data_loaders(data_dir="data", batch_size=512, test_split=0.2, random_state=42):
    """
    Create train and validation data loaders.
    
    Args:
        data_dir: Directory containing dataset
        batch_size: Batch size for training
        test_split: Fraction for validation split
        random_state: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataset
    X, Y = load_dataset(data_dir)
    
    print(f"Loaded dataset: X shape {X.shape}, Y shape {Y.shape}")
    
    # Train/validation split
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    Y_train, Y_val = Y[train_indices], Y[val_indices]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Create datasets
    train_dataset = HangmanDataset(X_train, Y_train)
    val_dataset = HangmanDataset(X_val, Y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, val_loader


def create_single_sample_loader(current_word: str):
    """
    Create a data loader for a single game state.
    
    Args:
        current_word: Current hangman word state (with underscores)
        
    Returns:
        DataLoader with single sample
    """
    from prepare.data import gen_row
    
    # Generate feature row
    X = gen_row(current_word)  # Shape: (1, 34)
    Y = np.zeros((1, 26), dtype=np.float32)  # Dummy labels
    
    dataset = HangmanDataset(X, Y)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return loader