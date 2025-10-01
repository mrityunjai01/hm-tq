import numpy as np
from numpy.typing import NDArray

from .combinations import sort_key
import torch


def gen_x_y_for_word(
    word: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # pyright: ignore[reportInvalidTypeArguments]
    """
    Generates masked word sequences for BERT-style training.
    Randomly masks at most 3 character types with '{' token.
    Returns: (masked_word_onehot, original_word_indices)
    """
    import random

    unique_chars = list(set(word))

    sorted_chars = sorted(unique_chars, key=sort_key)

    max_num_to_mask = min(6, random.randint(1, len(unique_chars)))
    exp_num_to_mask = min(3, random.randint(1, len(unique_chars)))
    chars_to_mask = []
    last_idx = len(sorted_chars) - 1
    second_p = 0.8
    p = exp_num_to_mask / len(sorted_chars) / second_p
    while len(chars_to_mask) < max_num_to_mask and last_idx >= 0:
        chars_to_mask.append(sorted_chars[last_idx])
        last_idx -= 1
        if random.random() < p:
            break
    actual_chars_to_mask = []
    for char in chars_to_mask:
        if random.random() < second_p:
            actual_chars_to_mask.append(char)

    # print(f"chars_to_mask = {chars_to_mask}")
    # print(f"actual_chars_to_mask = {actual_chars_to_mask}")

    if len(actual_chars_to_mask) == 0:
        actual_chars_to_mask.append(sorted_chars[-1])

    if len(actual_chars_to_mask) >= len(unique_chars) - 1:
        actual_chars_to_mask = actual_chars_to_mask[
            : -(max(0, len(actual_chars_to_mask) - (len(unique_chars) - 2)))
        ]

    masked_word = word
    for char in actual_chars_to_mask:
        masked_word = masked_word.replace(char, "{")

    masked_chars = [ord(c) - ord("a") for c in masked_word]
    original_chars = [ord(c) - ord("a") for c in word]
    mask = [c == "{" for c in masked_word]

    return (
        torch.tensor(masked_chars, dtype=torch.int32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(original_chars, dtype=torch.long),
    )


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
    word = "caterpillar"
    result = gen_x_y_for_word(word)
    x, mask, y = result

    print(f"x: {x}")
    print(f"mask: {mask}")
    print(f"y: {y}")
