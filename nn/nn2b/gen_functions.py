import itertools
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .combinations import create_combinations, sort_key
from .base_model import char_freq_values

unknown_row = char_freq_values + [0]
unknown_row = np.array(unknown_row, dtype=np.float32)
unknown_row /= unknown_row.sum()


def onehot(x: list[int]) -> NDArray[np.float32]:
    result = []
    # breakpoint()
    for w in x:
        if w > 26:
            result.append(unknown_row)

        else:
            a = np.zeros(27, dtype=np.float32)
            a[w] = 1
            result.append(a)

    return np.array(result, dtype=np.float32)


def gen_x_y_for_word(
    word: str, surr: int, pos_embed_size: int = 8
) -> tuple[tuple[NDArray[np.float32], NDArray[np.int32]], NDArray[np.int32]]:
    """
    Returns word-level sequences without padding for attention mechanism.
    Returns sequences of full words with position embeddings.
    """
    x_word_sequences: list[NDArray[np.float32]] = []
    x_positions: list[int] = []
    y: list[list[int]] = []
    
    original_word_len = len(word)
    
    # Convert word to one-hot without padding
    word_chars = [ord(c) - ord("a") for c in word]
    word_onehot = onehot(word_chars)
    
    # For each character position, create target
    for i in range(len(word)):
        position_fraction = i / original_word_len
        position_encoding = min(pos_embed_size - 1, int(position_fraction * pos_embed_size))
        
        x_word_sequences.append(word_onehot)
        x_positions.append(position_encoding)
        y.append([ord(word[i]) - ord("a")])

    set_chars = set(word)
    sorted_chars = sorted(list(set_chars), key=sort_key)

    for combination in create_combinations(sorted_chars):
        masked_word = word
        for ch in set_chars:
            if ch not in combination:
                masked_word = masked_word.replace(ch, "|")

        # Convert masked word to indices, replacing | with 27 (unknown)
        masked_chars = []
        for c in masked_word:
            if c == "|":
                masked_chars.append(27)  # Unknown token
            else:
                masked_chars.append(ord(c) - ord("a"))
        
        masked_onehot = onehot(masked_chars)
        
        for i in range(len(masked_word)):
            if masked_word[i] == "|":
                position_fraction = i / original_word_len
                position_encoding = min(pos_embed_size - 1, int(position_fraction * pos_embed_size))
                
                x_word_sequences.append(masked_onehot)
                x_positions.append(position_encoding)
                y.append([ord(word[i]) - ord("a")])

    x = (
        np.array(x_word_sequences, dtype=np.float32),
        np.array(x_positions, dtype=np.int32),
    )
    return x, np.array(y, dtype=np.int32)


def gen_x(word: str, surr: int) -> NDArray[np.int32]:
    """
    retired

    """

    x: list[list[int]] = []
    word = (
        "".join(["{" for _ in range(surr)]) + word + "".join(["{" for _ in range(surr)])
    )
    word = word.replace("_", "|")

    for i in range(surr, len(word) - surr):
        if word[i] == "|":
            x_row = [
                ord(w) - ord("a")
                for w in (word[i - surr : i] + word[i + 1 : i + surr + 1])
            ]
            x.append(x_row)
    return np.array(x, dtype=np.int32)


if __name__ == "__main__":
    result = gen_x_y_for_word("catez", 3)
    x, y = result
    assert isinstance(x, tuple)
    # breakpoint()
    assert x[0].shape[1] == 6
    assert x[0].shape[2] == 27
    assert x[1].shape[0] == x[0].shape[0] == y.shape[0]
