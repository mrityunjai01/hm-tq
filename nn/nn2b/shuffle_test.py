# read w_test and shuffle it and write it back
#
#
import random


def shuffle_test_words(test_words_file: str = "w_test.txt") -> None:
    with open(test_words_file, "r") as f:
        test_words = [w.strip() for w in f.readlines()]
    random.shuffle(test_words)
    with open(test_words_file, "w") as f:
        for word in test_words:
            f.write(word + "\n")


if __name__ == "__main__":
    shuffle_test_words("w_test.txt")

