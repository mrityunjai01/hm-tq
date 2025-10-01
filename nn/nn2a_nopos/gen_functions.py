import numpy as np
from numpy.typing import NDArray

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
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    returns a (combinations - 1) X 34 array, and a
    (combinations - 1) X 26 array of labels
    """
    x_surroundings: list[NDArray[np.float32]] = []
    y: list[list[int]] = []
    word = (
        "".join(["{" for _ in range(surr)]) + word + "".join(["{" for _ in range(surr)])
    )

    for i in range(surr, len(word) - surr):
        x_row = [
            ord(w) - ord("a") for w in (word[i - surr : i] + word[i + 1 : i + surr + 1])
        ]
        x_row = onehot(x_row)
        y_row = [ord(word[i]) - ord("a")]

        x_surroundings.append(x_row)
        y_row = [ord(word[i]) - ord("a")]

        y.append(y_row)

    x = np.array(x_surroundings, dtype=np.float32)
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
