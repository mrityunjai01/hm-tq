import os
import pickle

import numpy as np
import torch

from game.play_game import play_game
from .scrap import scrap_g
from tqdm import tqdm

from .model1 import HangmanNet
from .predict import beam_search_predict, predict

pass_words = []
words_lost = []

pass_filename = "pass.bin"


def save_pass_words():
    with open(pass_filename, "wb") as f:
        pickle.dump(pass_words, f)


def load_pass_words():
    global pass_words
    with open(pass_filename, "rb") as f:
        pass_words = pickle.load(f)


def create_model_guesser(model, verbose=False, surr: int = 3):
    """
    Create a guess function that uses the trained model to predict letters.

    Args:
        model: Trained CatBoost model
        max_guesses: Maximum number of wrong guesses allowed

    Returns:
        Guess function for use with play_game
    """

    def guess_fn(current_word: str, scrap=False) -> str:
        if not hasattr(guess_fn, "already_guessed"):
            guess_fn.already_guessed = set()  # pyright: ignore[reportFunctionMemberAccess]
            guess_fn.last_word = ""
            guess_fn.last_scrap = False
            guess_fn.cached_preds = None

        if (
            (current_word == guess_fn.last_word)
            and (guess_fn.cached_preds is not None)
            and ((scrap and guess_fn.last_scrap) or (not scrap))
        ):
            sorted_predictions = guess_fn.cached_preds
        else:
            if verbose:
                print(f"already guessed: {guess_fn.already_guessed}")

            if scrap:
                sorted_predictions = scrap_g(current_word)
                if sorted_predictions is None:
                    sorted_predictions = predict(
                        current_word,
                        model,
                        # already_guessed=guess_fn.already_guessed,
                        verbose=verbose,
                    )
                guess_fn.last_scrap = True
            else:
                sorted_predictions: list[int] = predict(
                    current_word,
                    model,
                    # already_guessed=guess_fn.already_guessed,
                    verbose=verbose,
                )
                guess_fn.last_scrap = False
            guess_fn.last_word = current_word
            guess_fn.cached_preds = sorted_predictions

        for idx in sorted_predictions:
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
    verbose=False,
) -> float:
    """
    Test the trained model by playing actual hangman games.

    Args:
        model_filepath: Path to the saved model
        test_words_file: Path to test words file
        max_test_words: Maximum number of words to test (for speed)
    """

    if not os.path.exists(test_words_file):
        print(f"Test words file {test_words_file} not found.")
        raise FileNotFoundError(f"Test words file {test_words_file} not found.")

    with open(test_words_file, "r") as f:
        test_words = [w.strip() for w in f.readlines()]

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

    for i, word in tqdm(enumerate(test_words)):
        result = model_single_game(
            test_word=word,
            model=model,
            verbose=False,
        )
        model_results.append(result)
        if verbose:
            if (i + 1) % 500 == 0:
                print(f"Played {i + 1}/{len(test_words)} games")

    model_win_rate = float(np.mean(model_results))
    if verbose:
        print(f"Model win rate: {model_win_rate:.2%}")

    return model_win_rate


def model_single_game(
    model_filepath: str = "models/cb.pth",
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
    global pass_words, words_lost

    assert model is not None

    model_guesser = create_model_guesser(
        model,
        verbose=verbose,
    )
    reset_guesser_state(model_guesser)
    model_result = play_game(test_word, model_guesser, verbose=verbose)
    if model_result == 0:
        words_lost.append(test_word)

    if model_result == 0 and test_word not in pass_words:
        pass_words.append(test_word)
        # print(f"Failed on word: {test_word}. Total failed words: {len(words_lost)}")
        # play = input("play?")
        # if play == "j":
        #     model_guesser = create_model_guesser(
        #         model,
        #         verbose=True,
        #     )
        #     play_game(test_word, model_guesser, verbose=True)
        # elif play == "p":
        #     pass_words.append(test_word)
        # elif play == "sync":
        #     save_pass_words()
        # elif play == "clear":
        #     pass_words = []

    return model_result


if __name__ == "__main__":
    device = "cpu"
    model = HangmanNet(vocab_size=27, device=device, num_layers=3).to(device)
    model = torch.compile(model, mode="max-autotune")

    model_filepath = "models/nn2b.pth_checkpoint_2"
    model.load_state_dict(
        torch.load(model_filepath, weights_only=True, map_location=torch.device(device))
    )
    test_model_on_game_play(model_object=model, verbose=True)
