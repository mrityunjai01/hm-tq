from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pandas as pd
import types
from collections import defaultdict
import itertools
from dataclasses import dataclass, field

with open("words_250000_train.txt", "r") as f:
    words: list[str] = [w.strip() for w in f.readlines()]


@dataclass
class State:
    already_guessed: set[str] = field(default_factory=set)

    def clear(self):
        self.already_guessed.clear()


def guess(self: State, word: str) -> str:
    letter_indices = list(range(97, 123))
    for letter_idx in letter_indices:
        if letter_idx not in self.already_guessed:
            self.already_guessed.append(letter_idx)
            return chr(letter_idx)


def play_game(actual_word, guess_fn) -> int:
    max_mistakes = 7
    current_word = "".join(["_"] * len(actual_word))
    mistakes = 0
    while mistakes < max_mistakes:
        guesssed_word = guess_fn(current_word)
        if not guesssed_word.isalpha() or len(guesssed_word) != 1:
            raise ValueError("Guess must be a single letter")
        if guesssed_word in actual_word:
            new_word = list(current_word)
            for i, c in enumerate(actual_word):
                if c == guesssed_word:
                    new_word[i] = guesssed_word
            current_word = "".join(new_word)
            if current_word == actual_word:
                return 1
        else:
            mistakes += 1
    return 0


state = State()
state.guess = types.MethodType(guess, state)
play_game("bhijkccnbr", lambda x: state.guess(x))
char_freq: defaultdict[str, int] = defaultdict(int)
for word in words:
    for l in word:
        char_freq[l] += 1

char_freq = dict(sorted(char_freq.items(), key=lambda item: item[1], reverse=True))
char_freq


def guess(self, word: str) -> str:
    letter_indices = list(range(97, 123))
    for letter_idx in letter_indices:
        if letter_idx not in self.already_guessed:
            self.already_guessed.add(letter_idx)
            return chr(letter_idx)


state = State()
state.guess = types.MethodType(guess, state)

game_results = []
for word in words:
    game_result = play_game(word, lambda x: state.guess(x))
    game_results.append(game_result)
    state.clear()


print("mean score:", np.mean(game_results))
freq_descending_list = list(char_freq.keys())


def guess(self, word: str) -> str:
    letter_indices = list(range(97, 123))
    for letter in freq_descending_list:
        if letter not in self.already_guessed:
            self.already_guessed.add(letter)
            return letter


state = State()
state.guess = types.MethodType(guess, state)

game_results = []
for word in words:
    game_result = play_game(word, lambda x: state.guess(x))
    game_results.append(game_result)
    state.clear()


print("mean score:", np.mean(game_results))

word_lens = [len(word) for word in words]
print(np.percentile(word_lens, 99))

max_revealed_chars = 17


def gen_x_y_for_word(word: str) -> tuple[np.array, np.array]:
    """
    returns a (combinations - 1) X 34 array, and a
    (combinations - 1) X 26 array of labels
    """
    set_chars = set(word)
    x = []
    y = []
    for comb_len in range(1, min(len(set_chars), max_revealed_chars)):
        for combination in itertools.combinations(set_chars, comb_len):
            x_row = [0] * 34
            y_row = [0] * 26
            x.append(x_row)
            y.append(y_row)
    return np.array(x), np.array(y)
