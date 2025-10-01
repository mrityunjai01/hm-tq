#!/usr/bin/env python3
"""
Training and testing script for hangman neural network model.
"""

import os
import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from nn.model import create_model
from nn.train import train_model
from nn.data_loader import create_data_loaders
from cb.cb_game_test import test_model_on_game_play
from prepare.data import gen_row
from stage import STAGE


class HangmanNetWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class to make HangmanNet compatible with sklearn interface.
    This allows us to reuse the game testing code from cb_game_test.py
    """

    def __init__(self, model, device="cpu", vocab_size=27):
        self.model = model
        self.device = device
        self.vocab_size = vocab_size

    def _preprocess_single(self, X):
        """Preprocess single sample for prediction."""
        # Convert to one-hot encoding
        X_shifted = X + 1  # -1 becomes 0, 0 becomes 1, ..., 26 becomes 27
        X_shifted = np.clip(X_shifted, 0, self.vocab_size - 1)

        batch_size, seq_len = X_shifted.shape
        X_onehot = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)

        for i in range(batch_size):
            for j in range(seq_len):
                X_onehot[i, j, X_shifted[i, j]] = 1.0

        return torch.FloatTensor(X_onehot).to(self.device)

    def predict_proba(self, X):
        """
        Predict class probabilities for each output label.

        Args:
            X: Input features (1, 34) from gen_row

        Returns:
            List of probability arrays for each label (compatible with CatBoost format)
        """
        self.model.eval()

        with torch.no_grad():
            X_tensor = self._preprocess_single(X)
            outputs = self.model(X_tensor)  # Shape: (1, 26)
            probs = outputs.cpu().numpy()[0]  # Shape: (26,)

            # Return in format expected by cb_game_test.py
            # Each element should be [prob_class_0, prob_class_1]
            result = []
            for prob in probs:
                result.append([[1 - prob, prob]])  # [P(class=0), P(class=1)]

            return result

    def predict(self, X):
        """Make binary predictions."""
        proba = self.predict_proba(X)
        # Convert to binary predictions (threshold at 0.5)
        return np.array([1 if p[0][1] > 0.5 else 0 for p in proba]).reshape(1, -1)


def main():
    """Main training and testing function."""
    print("=== Hangman Neural Network Training and Testing ===")

    # Hyperparameters
    if STAGE:
        batch_size = 64
        num_epochs = 5
        hidden_dim1 = 128
        hidden_dim2 = 64
        max_test_words = 50
    else:
        batch_size = 512
        num_epochs = 50
        hidden_dim1 = 512
        hidden_dim2 = 256
        max_test_words = None

    learning_rate = 0.001
    dropout_rate = 0.3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        batch_size=batch_size, test_split=0.2
    )

    # Model
    print("Creating model...")
    model = create_model(
        hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout_rate=dropout_rate
    )

    # Train
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path="models/hangman_net.pth",
        use_early_stopping=True,
        patience=10 if not STAGE else 3,
    )

    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Best F1 (micro): {max(history['val_f1_micro']):.6f}")
    print(f"Best F1 (macro): {max(history['val_f1_macro']):.6f}")

    # Test on game play
    print("\n=== Testing on Game Play ===")

    # Wrap model for compatibility with game testing
    wrapper = HangmanNetWrapper(model, device)

    # Test using the same interface as CatBoost
    try:
        test_score = test_model_on_game_play(
            model_filepath=None, model_object=wrapper, max_test_words=max_test_words
        )
        print(f"Neural Network Game Test Score: {test_score:.4f}")

    except Exception as e:
        print(f"Game testing failed: {e}")
        print(
            "You can test the model manually using the saved checkpoint at 'models/hangman_net.pth'"
        )

    # Save final results
    results = {
        "best_val_loss": min(history["val_loss"]),
        "best_f1_micro": max(history["val_f1_micro"]),
        "best_f1_macro": max(history["val_f1_macro"]),
        "game_test_score": test_score if "test_score" in locals() else None,
    }

    print("\n=== Final Results ===")
    for key, value in results.items():
        if value is not None:
            print(f"{key}: {value:.6f}")


def load_trained_model(checkpoint_path="models/hangman_net.pth", device="cpu"):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to saved model checkpoint
        device: Device to load model on

    Returns:
        Trained model and wrapper
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # Create model
    model = create_model()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create wrapper
    wrapper = HangmanNetWrapper(model, device)

    return model, wrapper


if __name__ == "__main__":
    main()

