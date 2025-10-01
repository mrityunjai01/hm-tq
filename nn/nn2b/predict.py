import numpy as np
import torch

from .base_model import char_freq, most_cooccuring_values


def predict(
    word: str,
    model,
) -> list[int]:
    """I will use model to predict a char from 'a' to 'z' for each '_'"""

    n_known_chars = set([ch for ch in word if ch != "_"])
    if len(n_known_chars) == 0:
        return [ord(c) - ord("a") for c in char_freq]
    elif len(n_known_chars) == 1:
        return [
            ord(c) - ord("a") for c in most_cooccuring_values[next(iter(n_known_chars))]
        ]

    blank_positions = [i for i, char in enumerate(word) if char == "_"]
    if not blank_positions:
        raise ValueError("No blanks in the word to predict.")

    char_indices = []
    for char in word:
        if char == "_":
            char_indices.append(26)  # '_' as unknown
        else:
            char_indices.append(ord(char) - ord("a"))
    x = torch.tensor(char_indices, dtype=torch.int32)
    predictions = model.predict_numpy(x).squeeze()

    at_least_one_pred = predictions[blank_positions].max(axis=0)
    return np.argsort(at_least_one_pred).tolist()[::-1]


def beam_search_predict(word, model, surr, k):
    original_word_len = len(word)
    blank_positions = [i for i, char in enumerate(word) if char == "_"]
    if not blank_positions:
        raise ValueError("No blanks in the word to predict.")

    padded_word = "{" * surr + word + "{" * surr
    beam = [(0.0, padded_word)]

    for pos_idx, original_pos in enumerate(blank_positions):
        pos = original_pos + surr  # Adjust for padding
        new_beam = []

        for score, hypothesis in beam:
            surrounding = (
                hypothesis[pos - surr : pos] + hypothesis[pos + 1 : pos + surr + 1]
            )

            char_indices = []
            for char in surrounding:
                if char == "{":
                    char_indices.append(26)  # '{' maps to 26
                elif char == "_":
                    char_indices.append(27)  # '_' as unknown
                else:
                    char_indices.append(ord(char) - ord("a"))

            x_surrounding = char_indices

            position_fraction = (pos - surr) / original_word_len
            position_encoding = min(7, int(position_fraction * 8))

            x_surrounding = np.expand_dims(x_surrounding, axis=0)  # Add batch dimension
            x_position = np.array([position_encoding], dtype=np.float32)

            predictions = model.predict_numpy((x_surrounding, x_position), verbose=0)
            predictions = predictions[0]
            breakpoint()

            # Get top k predictions
            top_k_indices = np.argsort(predictions)[-k:][::-1]

            for char_idx in top_k_indices:
                if char_idx < 26:  # Only consider a-z
                    char = chr(char_idx + ord("a"))
                    new_hypothesis = hypothesis[:pos] + char + hypothesis[pos + 1 :]
                    new_score = score + np.log(
                        predictions[char_idx] + 1e-8
                    )  # Add small epsilon to avoid log(0)
                    new_beam.append((new_score, new_hypothesis))

        # Keep top k hypotheses
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:k]

    # Remove padding and return top predictions
    results = []
    for score, hypothesis in beam:
        unpadded = hypothesis[surr:-surr]
        results.append(unpadded)

    return results
