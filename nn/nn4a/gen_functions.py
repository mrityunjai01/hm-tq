import itertools
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray


def gen_x_y_for_word(
    word: str, surr: int
) -> tuple[tuple[NDArray[np.int32], NDArray[np.int32]], NDArray[np.int32]]:
    """
    returns ((combinations - 1) X 34 array, (combinations - 1) array of positions), and a
    (combinations - 1) X 26 array of labels
    """
    x_surroundings: list[list[int]] = []
    x_positions: list[int] = []
    y: list[list[int]] = []
    original_word_len = len(word)
    word = (
        "".join(["{" for _ in range(surr)]) + word + "".join(["{" for _ in range(surr)])
    )

    for i in range(surr, len(word) - surr):
        x_row = [
            ord(w) - ord("a") for w in (word[i - surr : i] + word[i + 1 : i + surr + 1])
        ]
        # Add positional encoding: fraction of word progress, discretized to 0-7
        position_fraction = (i - surr) / original_word_len
        position_encoding = min(7, int(position_fraction * 8))

        x_surroundings.append(x_row)
        x_positions.append(position_encoding)
        y_row = [ord(word[i]) - ord("a")]

        y.append(y_row)

    x = (
        np.array(x_surroundings, dtype=np.int32),
        np.array(x_positions, dtype=np.int32),
    )
    return x, np.array(y, dtype=np.int32)


def gen_x(word: str, surr: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    returns (underscores X 34 array, underscores array of positions)
    """

    x_surroundings: list[list[int]] = []
    x_positions: list[int] = []
    original_word = word
    original_word_len = len(word.replace("_", ""))
    word = (
        "".join(["{" for _ in range(surr)]) + word + "".join(["{" for _ in range(surr)])
    )
    word = word.replace("_", "|")

    underscore_count = 0
    for i in range(surr, len(word) - surr):
        if word[i] == "|":
            x_row = [
                ord(w) - ord("a")
                for w in (word[i - surr : i] + word[i + 1 : i + surr + 1])
            ]
            # Calculate position based on original word position
            original_pos = i - surr
            position_fraction = original_pos / len(original_word)
            position_encoding = min(7, int(position_fraction * 8))

            x_surroundings.append(x_row)
            x_positions.append(position_encoding)
            underscore_count += 1

    return (
        np.array(x_surroundings, dtype=np.int32),
        np.array(x_positions, dtype=np.int32),
    )


if __name__ == "__main__":
    result = gen_x_y_for_word("catez", 3)
    print(result)
