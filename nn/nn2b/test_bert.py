#!/usr/bin/env python3

import torch
import numpy as np
from .model1 import HangmanNet
from .gen_functions import gen_x_y_for_word


def test_bert_model():
    """Test the BERT model with a simple example"""

    model = HangmanNet(
        vocab_size=27,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.3,
        max_seq_len=20,
        device="cpu",
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    test_word = "marmalite"
    print(f"Testing with word: '{test_word}'")

    masked_word, original_indices = gen_x_y_for_word(test_word)
    print(f"Masked word shape: {masked_word.shape}")
    print(f"Original indices: {original_indices}")

    masked_chars = []
    for i in range(masked_word.shape[0]):
        masked_chars.append(chr(masked_word[i] + ord("a")))

    print(f"Original word: {test_word}")
    print(f"Masked word:   {''.join(masked_chars)}")

    model.eval()
    with torch.no_grad():
        x_batch = (
            torch.tensor(masked_word).unsqueeze(0).long()
        )  # (1, seq_len, vocab_size)
        print(f"Input shape: {x_batch.shape}")

        output = model(x_batch)
        print(f"Output shape: {output.shape}")

        predictions = torch.softmax(output, dim=-1)
        predicted_chars = torch.argmax(predictions, dim=-1)

        print("Predictions:")
        for i, (orig_idx, pred_idx) in enumerate(
            zip(original_indices, predicted_chars[0])
        ):
            orig_char = chr(orig_idx + ord("a"))
            pred_char = chr(pred_idx.item() + ord("a"))  # pyright: ignore[reportArgumentType]
            mask_char = masked_chars[i]
            print(
                f"  Position {i}: original='{orig_char}', masked='{mask_char}', predicted='{pred_char}'"
            )


if __name__ == "__main__":
    test_bert_model()
