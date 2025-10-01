from collections import defaultdict

vowels = {"a", "e", "i", "o", "u", "y"}
vowel_indices = [ord(c) - ord("a") for c in vowels]


def read_dataset():
    with open("w_train.txt", "r") as f:
        words = [w.strip() for w in f.readlines()]
    return words


def vowel_count(word: str, cons_triad_counter: defaultdict[str, int]):
    vowel_index_set = set(vowel_indices)

    for i in range(len(word) - 2):
        if (
            ord(word[i]) - ord("a") not in vowel_index_set
            and ord(word[i + 1]) - ord("a") not in vowel_index_set
            and ord(word[i + 2]) - ord("a") not in vowel_index_set
        ):
            cons_triad_counter[word[i : i + 3]] += 1


def count_all_cons_triads(words):
    cons_triad_counter = defaultdict(int)
    for word in words:
        vowel_count(word, cons_triad_counter)
    return cons_triad_counter


def rare_triads():
    words = read_dataset()
    triads = count_all_cons_triads(words)
    triads = {triad for triad, count in triads.items() if count > 1}
    rare_triads = set()
    non_vowel_indices = [i for i in range(26) if chr(i + ord("a")) not in vowels]

    for i in non_vowel_indices:
        for j in non_vowel_indices:
            created_substr = "_" + chr(ord("a") + i) + chr(ord("a") + j)
            found = False
            for triad in triads:
                if triad[1:] == created_substr[1:]:
                    found = True
                    break
            if not found:
                rare_triads.add(created_substr)
            created_substr = chr(ord("a") + i) + "_" + chr(ord("a") + j)
            found = False
            for triad in triads:
                if triad[0] + triad[2] == created_substr[0] + created_substr[2]:
                    found = True
                    break
            if not found:
                rare_triads.add(created_substr)
            created_substr = chr(ord("a") + i) + chr(ord("a") + j) + "_"
            found = False
            for triad in triads:
                if triad[:2] == created_substr[:2]:
                    found = True
                    break
            if not found:
                rare_triads.add(created_substr)
            found = False

    for i in non_vowel_indices:
        created_substr = "_" + "_" + chr(ord("a") + i)
        found = False
        for triad in triads:
            if triad[2] == created_substr[2]:
                found = True
                break
        if not found:
            rare_triads.add(created_substr)
        created_substr = "_" + chr(ord("a") + i) + "_"
        found = False
        for triad in triads:
            if triad[1] == created_substr[1]:
                found = True
                break
        if not found:
            rare_triads.add(created_substr)
        created_substr = chr(ord("a") + i) + "_" + "_"
        found = False
        for triad in triads:
            if triad[0] == created_substr[0]:
                found = True
                break
        if not found:
            rare_triads.add(created_substr)

    return rare_triads


def promote_vowels(sorted_list: list[int]) -> list[int]:
    vowel_sublist = [ch for ch in sorted_list if ch in vowel_indices]
    non_vowel_sublist = [ch for ch in sorted_list if ch not in vowel_indices]
    return vowel_sublist + non_vowel_sublist


def check_vowels(
    word: str, already_guessed: set[str], rare_triads: set[str], sorted_list: list[int]
) -> list[int]:
    global vowels
    left_vowels = set(vowel_indices).difference(already_guessed)
    vowel_count = sum(1 for ch in vowels if ch in word)

    if len(left_vowels) > 2 or len(left_vowels) < 1:
        return sorted_list
    if vowel_count >= 4:
        return sorted_list

    for i in range(len(word) - 2):
        subword = word[i : i + 3]
        vowel_count = sum(1 for ch in vowels if ch in subword)
        blank_count = subword.count("_")
        if ((vowel_count == 0) and subword in rare_triads) or (blank_count == 3):
            return promote_vowels(sorted_list)

    for i in range(len(word) - 3):
        subword = word[i : i + 4]
        vowel_count = sum(1 for ch in vowels if ch in subword)
        blank_count = subword.count("_")
        if (vowel_count == 0) and (blank_count >= 1):
            return promote_vowels(sorted_list)
    return sorted_list


if __name__ == "__main__":
    rt = rare_triads()
    sorted_list = check_vowels("tal__l", set(["a", "e", "i", "o"]), rt, [9, 20])
    print(promote_vowels([4, 5, 4, 20, 3, 4]))

    # words = read_dataset()

    # triads = count_all_cons_triads(words)
    # print(f"Total unique consonant triads: {len(triads)}")
    # for triad, count in sorted(triads.items(), key=lambda x: x[1], reverse=True)[:-20]:
    #     print(f"{triad}: {count}")
