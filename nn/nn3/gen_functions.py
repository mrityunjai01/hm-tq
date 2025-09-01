import itertools
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_config_create_combinations(
    path: str = "config.json",
) -> tuple[
    int,
    int,
    list[int],
]:
    with open(path, "r") as f:
        config: dict[str, Any] = json.load(f)  # pyright: ignore[reportExplicitAny]
    return (
        config["char_cap_1"],
        config["char_cap_2"],
        config["char_blocks"],
    )


def load_config_sort_key(path="config.json") -> list[str]:
    with open(path, "r") as f:
        config = json.load(f)  # pyright: ignore[reportExplicitAny]
    return config["char_freq"]


def create_combinations(
    char_list: list[str],
    config=None,
) -> list[list[str]]:
    if config is None:
        char_cap_1, char_cap_2, char_blocks = load_config_create_combinations()
    else:
        char_cap_1, char_cap_2, char_blocks = (
            config["char_cap_1"],
            config["char_cap_2"],
            config["char_blocks"],
        )

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


char_freq = load_config_sort_key()


def sort_key(ch: str) -> int:
    return char_freq.index(ch) if ch in char_freq else 26


def gen_x_y_for_word(
    word: str, surr: int
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    returns a (combinations - 1) X 7 array, and a
    (combinations - 1) X 26 array of labels
    """
    set_chars = set(word)
    x: list[list[int]] = []
    y: list[list[int]] = []
    sorted_chars = sorted(list(set_chars), key=sort_key)

    for combination in create_combinations(sorted_chars):
        for ch in set_chars:
            if ch not in combination:
                newly_created_word = word.replace(ch, "|")

                newly_created_word = (
                    "".join(["{" for _ in range(surr)])
                    + word
                    + "".join(["{" for _ in range(surr)])
                )

                for i in range(surr, len(newly_created_word) - surr):
                    x_row = [
                        ord(w) - ord("a")
                        for w in (
                            newly_created_word[i - surr : i]
                            + newly_created_word[i + 1 : i + surr + 1]
                        )
                    ]
                    y_row = [ord(word[i - surr]) - ord("a")]

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
