#!/usr/bin/env python3
"""
Simple test script for neural network components.
"""

import torch
import numpy as np
from nn.model import create_model, count_parameters
from nn.data_loader import HangmanDataset
from prepare.data import gen_row


def test_model_creation():
    """Test model creation and forward pass."""
    print("=== Testing Model Creation ===")

    model = create_model()
    print(f"Model created successfully")
    print(f"Parameters: {count_parameters(model):,}")

    batch_size = 4
    seq_len = 34
    vocab_size = 27

    dummy_input = torch.randn(batch_size, seq_len, vocab_size)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    assert output.shape == (batch_size, 26), f"Expected (4, 26), got {output.shape}"
    assert torch.all((output >= 0) & (output <= 1)), "Output should be in [0, 1] range"

    print("✓ Model creation test passed")


def test_data_preprocessing():
    """Test data preprocessing and dataset creation."""
    print("\n=== Testing Data Preprocessing ===")

    X_dummy = np.array(
        [
            [
                1,
                5,
                -1,
                8,
                -1,
                12,
                5,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                5,
                12,
                -1,
                8,
                -1,
                5,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            [
                -1,
                8,
                5,
                12,
                12,
                15,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                15,
                12,
                12,
                5,
                8,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
        ],
        dtype=np.int32,
    )

    Y_dummy = np.array(
        [
            [
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ],
        dtype=np.int32,
    )

    dataset = HangmanDataset(X_dummy, Y_dummy)

    print(f"Dataset length: {len(dataset)}")

    x_sample, y_sample = dataset[0]
    print(f"Sample X shape: {x_sample.shape}")
    print(f"Sample Y shape: {y_sample.shape}")

    assert x_sample.shape == (34, 27), f"Expected (34, 27), got {x_sample.shape}"
    assert y_sample.shape == (26,), f"Expected (26,), got {y_sample.shape}"

    assert torch.allclose(x_sample.sum(dim=1), torch.ones(34)), (
        "Each position should have exactly one hot"
    )

    print("✓ Data preprocessing test passed")


def test_single_game_state():
    """Test preprocessing a single game state."""
    print("\n=== Testing Single Game State ===")

    current_word = "h_ll_"

    X_row = gen_row(current_word)
    print(f"Generated X row shape: {X_row.shape}")
    print(f"Generated X row: {X_row}")

    Y_dummy = np.zeros((1, 26), dtype=np.int32)
    dataset = HangmanDataset(X_row, Y_dummy)

    x_sample, y_sample = dataset[0]
    print(f"Processed sample shape: {x_sample.shape}")

    model = create_model()
    model.eval()

    with torch.no_grad():
        output = model(x_sample.unsqueeze(0))
        print(f"Model output shape: {output.shape}")
        print(f"Model output (first 10): {output[0, :10]}")

        probs = output[0].numpy()
        top_letters = np.argsort(probs)[-5:][::-1]
        print("Top 5 predicted letters:")
        for i, idx in enumerate(top_letters):
            letter = chr(ord("a") + idx)
            prob = probs[idx]
            print(f"  {i + 1}. {letter}: {prob:.4f}")

    print("✓ Single game state test passed")


def main():
    """Run all tests."""
    test_model_creation()
    test_data_preprocessing()
    test_single_game_state()

    print("\n=== All Tests Passed! ===")
    print("The neural network implementation is ready for training.")


if __name__ == "__main__":
    main()

