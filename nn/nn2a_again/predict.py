from numpy.typing import NDArray
from .gen_functions import unknown_row, onehot
import numpy as np

initial_eq_sweeps = 2
eq_k = 3
initial_sweeps = 2
target_vocab_size = 27  # a-z


def spot_predict(
    encoded_word: NDArray,
    model,
    pos: int,
    surr: int,
    already_guessed: set[int],
    eq_sweeps=True,
):
    surrounding = np.concatenate(
        [encoded_word[pos - surr : pos], encoded_word[pos + 1 : pos + surr + 1]],
        axis=0,
    )

    original_word_len = encoded_word.shape[0] - 2 * surr

    position_fraction = (pos - surr) / original_word_len
    position_encoding = min(7, int(position_fraction * 8))

    x_surrounding = np.expand_dims(surrounding, axis=0)  # Add batch dimension
    x_position = np.array([position_encoding], dtype=np.int32)

    predictions = model.predict_numpy(
        x_surrounding,
        x_position,
    ).flatten()
    predictions = np.append(predictions, 0)
    assert predictions.shape[0] == target_vocab_size
    for idx in already_guessed:
        predictions[idx] = 0.0  # Zero out already guessed characters
    if eq_sweeps:
        # set top 3 to equal probability
        top_k_indices = np.argsort(predictions)[-eq_k:]
        predictions[top_k_indices] = np.max(predictions)

    predictions /= predictions.sum()  # Re-normalize to sum to 1

    encoded_word[pos] = predictions


def predict(
    word: str,
    model,
    surr: int,
    already_guessed: set[int],
    k: int = 3,
) -> list[int]:
    """I will use model to predict a char from 'a' to 'z' for each '_'"""
    original_word_len = len(word)

    padded_word = "{" * surr + word + "{" * surr
    blank_positions = [i for i, char in enumerate(padded_word) if char == "_"]
    if not blank_positions:
        raise ValueError("No blanks in the word to predict.")

    char_indices = []
    for char in padded_word:
        if char == "_":
            char_indices.append(27)  # '_' as unknown
        else:
            char_indices.append(ord(char) - ord("a"))

    encoded_word = onehot(char_indices)

    for _ in range(initial_eq_sweeps):
        for pos in blank_positions:
            spot_predict(
                encoded_word, model, pos, surr, already_guessed, eq_sweeps=True
            )
    for _ in range(initial_sweeps):
        for pos in blank_positions:
            spot_predict(
                encoded_word, model, pos, surr, already_guessed, eq_sweeps=False
            )
    at_least_one_pred = 1 - (1 - encoded_word[blank_positions]).prod(axis=0)
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

            x_surrounding = onehot(char_indices)

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
