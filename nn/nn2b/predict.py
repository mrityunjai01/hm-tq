import numpy as np
import torch

from .base_model import char_freq, most_cooccuring_values


def predict(
    word: str,
    model,
    verbose=False,
) -> list[int]:
    """I will use model to predict a char from 'a' to 'z' for each '_'"""
    if verbose:
        breakpoint()

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

    # at_least_one_pred = predictions[blank_positions].max(axis=0)
    at_least_one_pred = 1 - (1 - predictions[blank_positions]).prod(axis=0)
    # at_least_one_pred = predictions[blank_positions].mean(axis=0)
    return np.argsort(at_least_one_pred).tolist()[::-1]


def single_shot(word, model):
    char_indices = []
    for char in word:
        if char == "_":
            char_indices.append(26)  # '_' as unknown
        else:
            char_indices.append(ord(char) - ord("a"))
    x = torch.tensor(char_indices, dtype=torch.int32)
    return model.predict_numpy(x).squeeze()


def beam_search_predict(word, model, verbose=False, already_guessed=set()):
    k = 2

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

    first_pos = blank_positions[0]
    variants = [(list(word), 1)]
    for pos in blank_positions:
        new_variants = []
        for variant in variants:
            predictions = single_shot("".join(variant[0]), model)[pos].tolist()
            for ag in already_guessed:
                predictions[ag] = 0.0
            predictions = [(v, i) for i, v in enumerate(predictions)]
            predictions = sorted(predictions, reverse=True)
            for i in range(k):
                changed_word = variant[0].copy()
                changed_word[pos] = chr(predictions[i][1] + ord("a"))
                new_variants += ((changed_word, variant[1] * predictions[i][0]),)

        variants = sorted(new_variants, key=lambda x: x[1], reverse=True)[:k]

    predictions = np.zeros((len(blank_positions), 26), dtype=np.float32)

    best_variant = variants[0][0]
    for i, pos in enumerate(blank_positions):
        c_word = best_variant.copy()
        c_word[pos] = "_"
        predictions[i] = single_shot("".join(c_word), model)[pos]
        predictions[i, list(already_guessed)] = 0.0

    at_least_one_pred = 1 - (1 - predictions).prod(axis=0)

    return np.argsort(at_least_one_pred).tolist()[::-1]
