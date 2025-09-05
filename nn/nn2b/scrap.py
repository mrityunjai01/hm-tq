import regex as re
import numpy as np


def load_words():
    with open("w_train.txt", "r") as f:
        return [line.strip() for line in f.readlines()]


words = load_words()


def scrap_g(word):
    span = 6
    blank_positions = [i for i in range(len(word)) if word[i] == "_"]
    word = word.replace("_", ".")
    # breakpoint()

    word_len = len(word)
    n_total_occ = 0
    occurences = np.zeros(26, dtype=np.float32)
    for pos in blank_positions:
        if word_len < span:
            break
        for start in range(word_len - span + 1):
            end = start + span
            if start <= pos < end:
                pattern = word[start:end]
                for w in words:
                    for match in re.finditer(pattern, w):
                        letter = w[match.start() + (pos - start)]
                        letter_idx = ord(letter) - ord("a")
                        occurences[letter_idx] += 1
                        n_total_occ += 1
    if n_total_occ < 3:
        return None
    return np.argsort(occurences).tolist()[::-1]


if __name__ == "__main__":
    original_word = "withered"
    print(f"original_word = {original_word}")
    predictions = scrap_g("withe_ed")

    if predictions is not None:
        predictions_converted_to_char = [chr(c + ord("a")) for c in predictions]
        print(predictions_converted_to_char)
