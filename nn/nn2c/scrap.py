import regex as re
import numpy as np

from nn.nn2c.hconfig import HConfig


def load_word_dictionary():
    with open("hangman_data/train_words.txt", "r") as f:
        return "4".join([line.strip() for line in f.readlines()])


word_dictionary = load_word_dictionary()


def scrap_g(word, hconfig: HConfig):
    if (word.count("_") >= len(word) - (hconfig.min_non_blanks - 1)) or (
        word.count("_") >= hconfig.max_blanks + 1
    ):
        return []

    spans = [3, 5, 6]
    blank_positions = [i for i in range(len(word)) if word[i] == "_"]

    total_occurences = np.zeros(26, dtype=np.float32)

    word_len = len(word)
    for span in spans:
        for pos in blank_positions:
            if word_len < span:
                break
            for start in range(word_len - span + 1):
                end = start + span
                if start <= pos < end:
                    pattern = word[start:end]
                    if pattern.count("_") > len(pattern) * 0.4:
                        continue
                    pattern = pattern.replace("_", "[a-z]")

                    occurences = np.zeros(26, dtype=np.float32)
                    n_total_occ = 0
                    for match in re.finditer(pattern, word_dictionary):
                        letter = word_dictionary[match.start() + pos - start]
                        letter_idx = ord(letter) - ord("a")
                        if letter_idx < 0:
                            breakpoint()
                        occurences[letter_idx] += 1
                        n_total_occ += 1
                    if n_total_occ >= hconfig.min_total_occ:
                        total_occurences = np.maximum(
                            total_occurences, (occurences - 0.5) / (n_total_occ + 0.5)
                        )
    total_occurences = np.power(total_occurences, 4)

    return [(v, i) for i, v in enumerate(total_occurences.tolist())]


if __name__ == "__main__":
    original_word = "withered"
    print(f"original_word = {original_word}")
    hconfig = HConfig()
    predictions = scrap_g(
        "withe_ed",
        hconfig,
    )
    predictions = [c for _, c in predictions]

    if predictions is not None:
        predictions_converted_to_char = [chr(c + ord("a")) for c in predictions]
        print(predictions_converted_to_char)
