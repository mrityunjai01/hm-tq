import regex as re
import numpy as np

from nn.nn2b.hconfig import HConfig


def load_word_dictionary():
    with open("w_train.txt", "r") as f:
        return "4".join([line.strip() for line in f.readlines()])


word_dictionary = load_word_dictionary()


def scrap_g(word, hconfig: HConfig):
    if (word.count("_") >= len(word) - (hconfig.min_non_blanks - 1)) or (
        word.count("_") >= hconfig.max_blanks + 1
    ):
        return []

    spans = [hconfig.span_start, hconfig.span_start + 1]
    blank_positions = [i for i in range(len(word)) if word[i] == "_"]

    word_len = len(word)
    n_total_occ = 0
    occurences = np.zeros(26, dtype=np.int32)
    for span in spans:
        for pos in blank_positions:
            if word_len < span:
                break
            for start in range(word_len - span + 1):
                end = start + span
                if start <= pos < end:
                    pattern = word[start:end]
                    pattern = pattern.replace("_", "[a-z]")

                    for match in re.finditer(pattern, word_dictionary):
                        letter = word_dictionary[match.start() + pos - start]
                        letter_idx = ord(letter) - ord("a")
                        if letter_idx < 0:
                            breakpoint()
                        occurences[letter_idx] += 1
                        n_total_occ += 1
    if n_total_occ < hconfig.min_total_occ:
        return []
    occurences = occurences.astype(np.float32)
    occurences /= n_total_occ
    return sorted([(v, i) for i, v in enumerate(occurences.tolist())], reverse=True)


if __name__ == "__main__":
    original_word = "withered"
    print(f"original_word = {original_word}")
    predictions = scrap_g("withe_ed")
    predictions = [c for _, c in predictions]

    if predictions is not None:
        predictions_converted_to_char = [chr(c + ord("a")) for c in predictions]
        print(predictions_converted_to_char)
