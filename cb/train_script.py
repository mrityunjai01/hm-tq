#!/usr/bin/env python3
"""
Training script for hangman CatBoost model.
"""

import os
from cb.train import train
from prepare.create_dataset import load_dataset


def main():
    """Train the CatBoost model on hangman dataset."""
    print("=== Hangman CatBoost Training ===")

    # Load dataset
    try:
        X, Y = load_dataset("data")
        print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")
    except FileNotFoundError:
        print("Dataset not found. Please run 'python prepare/create_dataset.py' first.")
        return

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Train model
    model_path = "models/cb.pth"
    print(f"Training model... (saving to {model_path})")

    train(X, Y, model_path, verbose=True)

    print("Training complete!")
    print(f"Model saved to: {model_path}")
    print("\nTo test the model, run: python cb/test.py")


if __name__ == "__main__":
    main()
