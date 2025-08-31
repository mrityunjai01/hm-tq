import json
from typing import Any
import numpy as np
import itertools

from numpy.typing import NDArray


def load_config(path: str = "config.json") -> tuple[int, int, list[int], list[str]]:
    with open(path, "r") as f:
        config: dict[str, Any] = json.load(f)  # pyright: ignore[reportExplicitAny]
    return (
        config["char_cap_1"],
        config["char_cap_2"],
        config["char_blocks"],
        config["char_freq"],
    )


char_cap_1, char_cap_2, char_blocks, char_freq = load_config()


def create_combinations(char_list: list[str]) -> list[list[str]]:
    if len(char_list) > char_cap_2:
        return create_combinations(char_list[:char_cap_2])
    elif len(char_list) <= char_cap_1:
        combinations: list[list[str]] = []
        for combination_len in range(0, len(char_list)):
            combinations.extend(
                [
                    list(comb)
                    for comb in itertools.combinations(char_list, combination_len)
                ]
            )
        return combinations
    else:
        prev_char_block_sum = 0
        adapted_char_blocks = []
        for char_block_len in char_blocks:
            if prev_char_block_sum >= len(char_list):
                break

            adapted_char_blocks.append(
                min(char_block_len, len(char_list) - prev_char_block_sum)
            )
            prev_char_block_sum += char_block_len

        combinations = []
        prev_blocks_sum = 0

        for idx, char_block_len in enumerate(adapted_char_blocks):
            if idx == (len(adapted_char_blocks) - 1):
                last_block = True
            else:
                last_block = False

            for combination_len in range(0, char_block_len):  # pyright: ignore[reportArgumentType]
                for combination in itertools.combinations(
                    char_list[prev_blocks_sum : (prev_blocks_sum + char_block_len)],
                    combination_len,
                ):
                    combinations.append(
                        list(char_list[:prev_blocks_sum]) + list(combination)
                    )
            prev_blocks_sum += char_block_len
    return combinations


def sort_key(ch: str) -> int:
    return char_freq.index(ch) if ch in char_freq else 26


def gen_x_y_for_word(word: str) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    returns a (combinations - 1) X 34 array, and a
    (combinations - 1) X 26 array of labels
    """
    set_chars = set(word)
    x: list[list[int]] = []
    y: list[list[int]] = []
    sorted_chars = sorted(list(set_chars), key=sort_key)

    for combination in create_combinations(sorted_chars):
        x_row = [0 for _ in range(34)]
        y_row = [0 for _ in range(26)]
        for i in range(min(len(word), 17)):
            if word[i] in combination:
                x_row[i] = ord(word[i]) - ord("a") + 1
            else:
                x_row[i] = -1

            if word[-1 - i] in combination:
                x_row[33 - i] = ord(word[-1 - i]) - ord("a") + 1
            else:
                x_row[33 - i] = -1

        unrevealed_chars = set_chars.difference(set(combination))
        for ch in unrevealed_chars:
            y_row[ord(ch) - ord("a")] = 1

        x.append(x_row)
        y.append(y_row)
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def gen_row(word: str) -> NDArray[np.int32]:
    """
    returns a 1 X 34 array
    """

    x_row = [0 for _ in range(34)]
    for i in range(min(len(word), 17)):
        if word[i] != "_":
            x_row[i] = ord(word[i]) - ord("a") + 1
        else:
            x_row[i] = -1

        if word[-1 - i] != "_":
            x_row[33 - i] = ord(word[-1 - i]) - ord("a") + 1
        else:
            x_row[33 - i] = -1
    return np.array(x_row, dtype=np.int32).reshape(1, -1)
