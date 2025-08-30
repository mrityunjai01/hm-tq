import os

from prepare.separate_train_test import separate_train_test
from prepare.data import gen_x_y_for_word

if __name__ == "__main__":
    if not os.path.exists("w_train.txt") or not os.path.exists("w_test.txt"):
        separate_train_test()

    with open("w_train.txt", "r") as f:
        words = [w.strip() for w in f.readlines()]
