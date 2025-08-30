import os
from collections import defaultdict

from prepare.separate_train_test import separate_train_test
from prepare.data import gen_x_y_for_word

if __name__ == "__main__":
    x, y = gen_x_y_for_word("abcdefgh")
    print(x)
    print(y)
