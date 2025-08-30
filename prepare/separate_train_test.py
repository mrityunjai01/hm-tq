import numpy as np
from numpy.typing import NDArray


def write_list_to_file(lst: list[str], filename: str):
    with open(filename, "w") as f:
        for item in lst:
            _ = f.write(f"{item}\n")


def separate_train_test():
    np.random.seed(42)
    with open("words_250000_train.txt", "r") as f:
        words: list[str] = [w.strip() for w in f.readlines()]
    # choose 3k words for test, and the rest for train, shuffle then index
    shuffled_indices: NDArray[np.int64] = np.random.permutation(len(words))
    test_indices: NDArray[np.int64] = shuffled_indices[:3000]
    test_words: list[str] = [words[i] for i in test_indices]  # pyright: ignore[reportAny]
    write_list_to_file(test_words, "w_test.txt")
    train_indices = shuffled_indices[3000:]
    train_words: list[str] = [words[i] for i in train_indices]  # pyright: ignore[reportAny]
    write_list_to_file(train_words, "w_train.txt")
