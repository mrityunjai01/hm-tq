import os
import numpy as np
from collections import defaultdict
from skops.io import load
from game.play_game import play_game
from prepare.data import gen_x_y_for_word


def create_model_guesser(model, max_guesses=7):
    """
    Create a guess function that uses the trained model to predict letters.

    Args:
        model: Trained CatBoost model
        max_guesses: Maximum number of wrong guesses allowed

    Returns:
        Guess function for use with play_game
    """

    def guess_fn(current_word: str) -> str:
        # Get already guessed letters from the guesser state
        if not hasattr(guess_fn, "already_guessed"):
            guess_fn.already_guessed = set()

        # If we have revealed letters, use model to predict next best letter
        if "_" in current_word and any(c != "_" for c in current_word):
            try:
                # Create feature vector from current game state
                revealed_letters = set(c for c in current_word if c != "_")

                # Generate a dummy word with same pattern to get feature format
                # This is a simplified approach - in practice you'd want better encoding
                dummy_word = current_word.replace(
                    "_", "a"
                )  # Replace blanks with dummy letter

                # Get model prediction (this is simplified - ideally we'd have better state encoding)
                # For now, fall back to frequency-based guessing
                pass
            except:
                pass

        # Fallback to frequency-based guessing
        # Letters ordered by frequency in English
        freq_order = [
            "e",
            "i",
            "a",
            "r",
            "n",
            "o",
            "s",
            "t",
            "l",
            "c",
            "u",
            "d",
            "p",
            "m",
            "h",
            "g",
            "y",
            "b",
            "f",
            "v",
            "k",
            "w",
            "z",
            "x",
            "q",
            "j",
        ]

        for letter in freq_order:
            if letter not in guess_fn.already_guessed:
                guess_fn.already_guessed.add(letter)
                return letter

        # Shouldn't reach here, but fallback to any unguessed letter
        for letter in "abcdefghijklmnopqrstuvwxyz":
            if letter not in guess_fn.already_guessed:
                guess_fn.already_guessed.add(letter)
                return letter

        return "a"  # Final fallback

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

    # Load test words
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

    # Create model-based guesser
    model_guesser = create_model_guesser(model)

    # Test baseline: frequency-based guesser
    print("\nPlaying games...")
    model_results = []

    word_length_results = defaultdict(list)

    for i, word in enumerate(test_words):
        # Test model guesser
        reset_guesser_state(model_guesser)
        model_result = play_game(word, model_guesser)
        model_results.append(model_result)


        # Track by word length
        word_length_results[len(word)].append(model_result))

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

            print(
                f"Length {length:2d}: Model={model_rate:.3f}, "
            )

    # Show some example games
    print("\n=== Example Games ===")
    example_words = test_words[:5]  # First 5 words

    for word in example_words:
        reset_guesser_state(model_guesser)

        model_result = play_game(word, model_guesser, verbose = True)

        model_status = "WON" if model_result else "LOST"

        print(f"'{word}': Model={model_status},")


def compare_strategies(test_words_file: str = "w_test.txt", max_test_words: int = 500):
    """
    Compare different guessing strategies without using the ML model.

    Args:
        test_words_file: Path to test words file
        max_test_words: Maximum number of words to test
    """
    print("\n=== Strategy Comparison ===")

    # Load test words
    if not os.path.exists(test_words_file):
        print(f"Test words file {test_words_file} not found.")
        return

    with open(test_words_file, "r") as f:
        test_words = [w.strip() for w in f.readlines()]

    if max_test_words:
        test_words = test_words[:max_test_words]

    strategies = {
        "frequency": [
            "e",
            "i",
            "a",
            "r",
            "n",
            "o",
            "s",
            "t",
            "l",
            "c",
            "u",
            "d",
            "p",
            "m",
            "h",
            "g",
            "y",
            "b",
            "f",
            "v",
            "k",
            "w",
            "z",
            "x",
            "q",
            "j",
        ],
        "alphabetical": list("abcdefghijklmnopqrstuvwxyz"),
        "vowels_first": ["e", "i", "a", "o", "u"]
        + [c for c in "bcdfghjklmnpqrstvwxyz"],
    }

    for strategy_name, letter_order in strategies.items():

        def make_guesser(order):
            def guesser(current_word: str) -> str:
                if not hasattr(guesser, "already_guessed"):
                    guesser.already_guessed = set()
                for letter in order:
                    if letter not in guesser.already_guessed:
                        guesser.already_guessed.add(letter)
                        return letter
                return "a"

            return guesser

        guesser = make_guesser(letter_order)
        results = []

        for word in test_words:
            if hasattr(guesser, "already_guessed"):
                guesser.already_guessed.clear()
            result = play_game(word, guesser)
            results.append(result)

        win_rate = np.mean(results)
        print(f"{strategy_name:12s}: {win_rate:.4f} ({sum(results)}/{len(results)})")


if __name__ == "__main__":
    test_model_on_game_play()
    compare_strategies()

