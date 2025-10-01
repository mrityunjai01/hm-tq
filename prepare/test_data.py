import os
import json
import tempfile
from prepare.data import create_combinations, load_config


def test_load_config_reads_values_correctly():
    test_config = {
        "char_cap_1": 5,
        "char_cap_2": 10,
        "char_blocks": ["a", "b", "c"],
        "char_freq": {"a": 1, "b": 2},
    }
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        json.dump(test_config, tmp)
        tmp_path = tmp.name

    result = load_config(tmp_path)
    assert result == (5, 10, ["a", "b", "c"], {"a": 1, "b": 2})

    os.remove(tmp_path)


def test_create_combinations_empty():
    empty_list: list[str] = []
    assert create_combinations(empty_list) == []


def compare_sets(list1: list[list[str]], list2: list[list[str]]):
    list1_sorted = ["".join(sorted(sublist)) for sublist in list1]
    list2_sorted = ["".join(sorted(sublist)) for sublist in list2]

    if sorted(list1_sorted) != sorted(list2_sorted):
        print("Differences found:")
        print(
            "In list1_sorted but not in list2_sorted:",
            set(list1_sorted) - set(list2_sorted),
        )
        print(
            "In list2_sorted but not in list1_sorted:",
            set(list2_sorted) - set(list1_sorted),
        )

    return sorted(list1_sorted) == sorted(list2_sorted)


def test_create_combinations_below_cap_1():
    char_list = ["a", "b", "c"]
    combs = create_combinations(char_list)
    expected_combinations = [
        [],
        ["a"],
        ["b"],
        ["c"],
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
    ]
    assert len(combs) == len(expected_combinations)  # combinations of lengths 0, 1, 2

    assert compare_sets(combs, expected_combinations)


def test_create_combinations_larger_cap_1():
    char_list = ["a", "b", "c", "d", "e", "f", "g"]
    combs = create_combinations(char_list)
    expected_combinations = [
        [],
        ["a"],
        ["b"],
        ["c"],
        ["d"],
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
        ["a", "d"],
        ["b", "d"],
        ["c", "d"],
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "c", "d"],
        ["b", "c", "d"],
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", "d", "f"],
        ["a", "b", "c", "d", "g"],
        ["a", "b", "c", "d", "e", "f"],
        ["a", "b", "c", "d", "e", "g"],
        ["a", "b", "c", "d", "f", "g"],
    ]

    assert len(combs) == len(expected_combinations)  # combinations of lengths 0, 1, 2
    assert compare_sets(combs, expected_combinations)


def test_create_combinations_larger_cap_2():
    char_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    combs = create_combinations(char_list)
    expected_combinations = [
        [],
        ["a"],
        ["b"],
        ["c"],
        ["d"],
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
        ["a", "d"],
        ["b", "d"],
        ["c", "d"],
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "c", "d"],
        ["b", "c", "d"],
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", "d", "f"],
        ["a", "b", "c", "d", "g"],
        ["a", "b", "c", "d", "h"],
        ["a", "b", "c", "d", "e", "f"],
        ["a", "b", "c", "d", "e", "g"],
        ["a", "b", "c", "d", "e", "h"],
        ["a", "b", "c", "d", "f", "g"],
        ["a", "b", "c", "d", "f", "h"],
        ["a", "b", "c", "d", "g", "h"],
        ["a", "b", "c", "d", "e", "f", "g"],
        ["a", "b", "c", "d", "e", "f", "h"],
        ["a", "b", "c", "d", "e", "g", "h"],
        ["a", "b", "c", "d", "f", "g", "h"],
        ["a", "b", "c", "d", "e", "f", "g", "h"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "j"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "k"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "l"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "l"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "j", "l"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "k", "l"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l"],
    ]

    assert len(combs) == len(expected_combinations)  # combinations of lengths 0, 1, 2
    assert compare_sets(combs, expected_combinations)
