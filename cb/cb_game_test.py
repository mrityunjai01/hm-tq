import os
import numpy as np
from collections import defaultdict
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from skops.io import load
from game.play_game import play_game
from prepare.data import gen_row, gen_x_y_for_word


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
            guess_fn.already_guessed = set()

        predictions: NDArray[np.int32] = model.predict(gen_row(current_word))
        sorted_predictions = np.sort(predictions)[::-1]
        for pred in sorted_predictions:
            if pred not in guess_fn.already_guessed:
                guess_fn.already_guessed.add(pred)
                return chr(pred + ord("a") - 1)
        raise ValueError("No valid guesses left")

    return guess_fn


def reset_guesser_state(guess_fn):
    """Reset the guesser's state for a new game."""
    if hasattr(guess_fn, "already_guessed"):
        guess_fn.already_guessed.clear()


def test_model_on_game_play(
    model_filepath: str = "models/cb.pth",
    test_words_file: str = "w_test.txt",
    max_test_words: int | None = None,
) -> None:
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
        return

    with open(test_words_file, "r") as f:
        test_words = [w.strip() for w in f.readlines()]

    if max_test_words and len(test_words) > max_test_words:
        test_words = test_words[:max_test_words]
        print(f"Testing on first {max_test_words} words for speed")

    print(f"Testing on {len(test_words)} words...")

    # Load model
    if not os.path.exists(model_filepath):
        print(f"Model not found at {model_filepath}. Please train first.")
        return

    print(f"Loading model from {model_filepath}...")
    model = load(model_filepath, trusted=["catboost.core.CatBoostClassifier"])

    model_guesser = create_model_guesser(model)

    print("\nPlaying games...")
    model_results = []

    word_length_results = defaultdict(list)

    for i, word in enumerate(test_words):
        # Test model guesser
        reset_guesser_state(model_guesser)
        model_result = play_game(word, model_guesser)
        model_results.append(model_result)

        # Track by word length
        word_length_results[len(word)].append(model_result)

        if (i + 1) % 100 == 0:
            print(f"Played {i + 1}/{len(test_words)} games")

    # Calculate metrics
    print("\n=== Game Play Results ===")

    model_win_rate = np.mean(model_results)

    print(
        f"Model win rate: {model_win_rate:.4f} ({sum(model_results)}/{len(model_results)})"
    )

    # Results by word length
    print("\n=== Results by Word Length ===")
    for length in sorted(word_length_results.keys()):
        results = word_length_results[length]
        if len(results) >= 10:  # Only show if we have enough samples
            model_wins = [r[0] for r in results]

            model_rate = np.mean(model_wins)

            print(f"Length {length:2d}: Model={model_rate:.3f}, ")

    # Show some example games
    print("\n=== Example Games ===")
    example_words = test_words[:5]  # First 5 words

    for word in example_words:
        reset_guesser_state(model_guesser)

        model_result = play_game(word, model_guesser, verbose=True)

        model_status = "WON" if model_result else "LOST"

        print(f"'{word}': Model={model_status},")


def model_single_game(
    model_filepath: str = "models/cb.pth",
    test_word: str = "hangman",
) -> None:
    """
    Play a single game using the trained model.
    Args:
        model_filepath: Path to the saved model
        test_word: The word to guess in the game
    """

    if not os.path.exists(model_filepath):
        print(f"Model not found at {model_filepath}. Please train first.")
        return
    model = load(model_filepath, trusted=["catboost.core.CatBoostClassifier"])
    model_guesser = create_model_guesser(model, verbose=True)
    reset_guesser_state(model_guesser)
    model_result = play_game(test_word, model_guesser, verbose=True)
    return model_result


if __name__ == "__main__":
    result = model_single_game("ponytail")
    print(f"result of model game: {result}")
    # test_model_on_game_play()
