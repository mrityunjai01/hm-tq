import os

import numpy as np

from game.play_game import play_game
from .model1 import HangmanNet
from .predict import predict


def create_model_guesser(model, verbose=False, surr: int = 3):
    """
    Create a guess function that uses the trained model to predict letters.

    Args:
        model: Trained CatBoost model
        max_guesses: Maximum number of wrong guesses allowed

    Returns:
        Guess function for use with play_game
    """

    def guess_fn(current_word: str, actual_word: str) -> str:
        if not hasattr(guess_fn, "already_guessed"):
            guess_fn.already_guessed = set()  # pyright: ignore[reportFunctionMemberAccess]

        if verbose:
            print(f"already guessed: {guess_fn.already_guessed}")

        sorted_predictions: list[tuple[float, int]] = predict(
            current_word,
            model,
        )

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
    model_object=None,
    test_words_file: str = "w_test.txt",
    max_test_words: int | None = None,
    small_words=False,
    verbose=False,
) -> float:
    """
    Test the trained model by playing actual hangman games.

    Args:
        model_filepath: Path to the saved model
        test_words_file: Path to test words file
        max_test_words: Maximum number of words to test (for speed)
    """

    if verbose:
        print("=== CatBoost Model Game Play Evaluation ===")

    if not os.path.exists(test_words_file):
        print(f"Test words file {test_words_file} not found.")
        raise FileNotFoundError(f"Test words file {test_words_file} not found.")

    with open(test_words_file, "r") as f:
        test_words = [w.strip() for w in f.readlines()]

    if small_words:
        test_words = [w for w in test_words if len(w) <= 5]

    if max_test_words and len(test_words) > max_test_words:
        test_words = test_words[:max_test_words]
        print(f"Testing on first {max_test_words} words for speed")

    if verbose:
        print(f"Testing on {len(test_words)} words...")

    if model_object is not None:
        model = model_object
    else:
        raise ValueError("Either model_filepath or model_object must be provided.")
    model_results: list[int] = []

    for i, word in enumerate(test_words):
        result = model_single_game(
            test_word=word,
            model=model,
            verbose=False,
        )
        model_results.append(result)
        if verbose:
            if (i + 1) % 100 == 0:
                print(f"Played {i + 1}/{len(test_words)} games")

    model_win_rate = float(np.mean(model_results))
    return model_win_rate


def model_single_game(
    model=None,
    test_word: str = "hangman",
    verbose=False,
) -> int:
    """
    Play a single game using the trained model.
    Args:
        model_filepath: Path to the saved model
        test_word: The word to guess in the game
    """

    assert model is not None

    model_guesser = create_model_guesser(
        model,
        verbose=verbose,
    )
    reset_guesser_state(model_guesser)
    model_result = play_game(test_word, model_guesser, verbose=verbose)
    return model_result


if __name__ == "__main__":
    device = "cpu"
    model = HangmanNet(vocab_size=27, device=device, num_layers=2).to(device)
    model_filepath = "nn2b_gc_27.pth_checkpoint_27"
    model.load_state_dict(torch.load(model_filepath, weights_only=True))
    torch.load(model_filepath, map_location=device)
    test_model_on_game_play()
