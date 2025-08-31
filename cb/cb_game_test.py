import os

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin  # pyright: ignore[reportMissingTypeStubs]
from skops.io import load  # pyright: ignore[reportMissingTypeStubs]

from game.play_game import play_game
from prepare.data import gen_row


def create_model_guesser(model: ClassifierMixin, max_guesses=7, verbose=False):
    """
    Create a guess function that uses the trained model to predict letters.

    Args:
        model: Trained CatBoost model
        max_guesses: Maximum number of wrong guesses allowed

    Returns:
        Guess function for use with play_game
    """

    def guess_fn(current_word: str) -> str:
        if not hasattr(guess_fn, "already_guessed"):
            guess_fn.already_guessed = set()  # pyright: ignore[reportFunctionMemberAccess]

        raw_predictions: NDArray[np.float32] = model.predict_proba(
            gen_row(current_word)
        )
        predictions = [r[0][1] for r in raw_predictions]

        sorted_predictions = sorted(
            [(pred, idx) for idx, pred in enumerate(predictions)]
        )[::-1]

        if verbose:
            print(f"already guessed: {guess_fn.already_guessed}")

        for _, idx in sorted_predictions:
            if idx not in guess_fn.already_guessed:
                guess_fn.already_guessed.add(idx)
                return chr(idx + ord("a"))
        raise ValueError("No valid guesses left")

    return guess_fn


def reset_guesser_state(guess_fn):
    """Reset the guesser's state for a new game."""
    if hasattr(guess_fn, "already_guessed"):
        guess_fn.already_guessed.clear()


def test_model_on_game_play(
    model_filepath: str | None = "models/cb.pth",
    model_object: ClassifierMixin | None = None,
    test_words_file: str = "w_test.txt",
    max_test_words: int | None = None,
) -> float:
    """
    Test the trained model by playing actual hangman games.

    Args:
        model_filepath: Path to the saved model
        test_words_file: Path to test words file
        max_test_words: Maximum number of words to test (for speed)
    """

    print("=== CatBoost Model Game Play Evaluation ===")

    if not os.path.exists(test_words_file):
        print(f"Test words file {test_words_file} not found.")
        raise FileNotFoundError(f"Test words file {test_words_file} not found.")

    with open(test_words_file, "r") as f:
        test_words = [w.strip() for w in f.readlines()]

    if max_test_words and len(test_words) > max_test_words:
        test_words = test_words[:max_test_words]
        print(f"Testing on first {max_test_words} words for speed")

    print(f"Testing on {len(test_words)} words...")

    if model_object is not None:
        model = model_object
    else:
        if not os.path.exists(model_filepath):
            print(f"Model not found at {model_filepath}. Please train first.")
            raise FileNotFoundError(f"Model not found at {model_filepath}")

        model = load(model_filepath, trusted=["catboost.core.CatBoostClassifier"])
    model_results: list[int] = []

    for i, word in enumerate(test_words):
        result = model_single_game(test_word=word, model=model, verbose=False)
        model_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"Played {i + 1}/{len(test_words)} games")

    model_win_rate = float(np.mean(model_results))
    return model_win_rate

    # print(
    #     f"Model win rate: {model_win_rate:.4f} ({sum(model_results)}/{len(model_results)})"
    # )
    #
    # # Results by word length
    # print("\n=== Results by Word Length ===")
    # for length in sorted(word_length_results.keys()):
    #     results = word_length_results[length]
    #     if len(results) >= 10:  # Only show if we have enough samples
    #         model_wins = [r[0] for r in results]
    #
    #         model_rate = np.mean(model_wins)
    #
    #         print(f"Length {length:2d}: Model={model_rate:.3f}, ")
    #
    # # Show some example games
    # print("\n=== Example Games ===")
    # example_words = test_words[:5]  # First 5 words
    #
    # for word in example_words:
    #     reset_guesser_state(model_guesser)
    #
    #     model_result = play_game(word, model_guesser, verbose=True)
    #
    #     model_status = "WON" if model_result else "LOST"
    #
    #     print(f"'{word}': Model={model_status},")


def model_single_game(
    model_filepath: str = "models/cb.pth",
    model: ClassifierMixin | None = None,
    test_word: str = "hangman",
    verbose=False,
) -> int:
    """
    Play a single game using the trained model.
    Args:
        model_filepath: Path to the saved model
        test_word: The word to guess in the game
    """
    if model is None:
        if not os.path.exists(model_filepath):
            print(f"Model not found at {model_filepath}. Please train first.")
            raise FileNotFoundError(f"Model not found at {model_filepath}")

        model = load(model_filepath, trusted=["catboost.core.CatBoostClassifier"])

    assert model is not None

    model_guesser = create_model_guesser(model, verbose=verbose)
    reset_guesser_state(model_guesser)
    model_result = play_game(test_word, model_guesser, verbose=verbose)
    return model_result


if __name__ == "__main__":
    # result = model_single_game(verbose=True, test_word="example")
    # result = (verbose=True, test_word="example")
    # print(f"result of model game: {result}")
    test_model_on_game_play()
