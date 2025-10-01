import itertools
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray


def gen_x_y_for_word(
    word: str, surr: int
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    returns a (combinations - 1) X 34 array, and a
    (combinations - 1) X 26 array of labels
    """
    x: list[list[int]] = []
    y: list[list[int]] = []
    word = (
        "".join(["{" for _ in range(surr)]) + word + "".join(["{" for _ in range(surr)])
    )

    for i in range(surr, len(word) - surr):
        x_row = [
            ord(w) - ord("a") for w in (word[i - surr : i] + word[i + 1 : i + surr + 1])
        ]
        y_row = [ord(word[i]) - ord("a")]

        x.append(x_row)
        y.append(y_row)
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def gen_x(word: str, surr: int) -> NDArray[np.int32]:
    """
    returns a underscores X 34 array
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
    print(result)
