import os
import pickle

import numpy as np
import torch

from game.play_game import play_game
from nn.nn2b.hconfig import HConfig
from nn.nn2b.selector import SelectorNN, append_to_file
from nn.nn2b.validate import validate_predictions
from nn.nn2b.vowels import check_vowels, rare_triads
from .scrap import scrap_g
from tqdm import tqdm

from .model1 import HangmanNet
from .predict import predict

pass_words = []
words_lost = []

pass_filename = "pass.bin"
train_phase = False

selector_model_obj = SelectorNN(input_dim=13, hidden_dim=48, output_cardinality=6)
selector_model_obj.load_state_dict(
    torch.load(
        "models/selector_nn.pth",
        weights_only=True,
    )
)


def selector_model(x: torch.Tensor, targets: list[int]) -> list[tuple[float, int]]:
    selector_model_obj.eval()
    with torch.no_grad():
        output = selector_model_obj(x).numpy()

    return sorted(list(zip(output.flatten().tolist(), targets)), reverse=True)


def save_pass_words():
    with open(pass_filename, "wb") as f:
        pickle.dump(pass_words, f)


def load_pass_words():
    global pass_words
    with open(pass_filename, "rb") as f:
        pass_words = pickle.load(f)


rare_triads_set = rare_triads()


def create_model_guesser(
    model,
    hconfig: HConfig,
    verbose=False,
    small_words_model=None,
    surr: int = 3,
):
    """
    Create a guess function that uses the trained model to predict letters.

    Args:
        model: Trained CatBoost model
        max_guesses: Maximum number of wrong guesses allowed

    Returns:
        Guess function for use with play_game
    """

    def guess_fn(current_word: str, correct_word: str) -> str:
        if not hasattr(guess_fn, "already_guessed"):
            guess_fn.already_guessed = set()  # pyright: ignore[reportFunctionMemberAccess]
            guess_fn.last_word = ""
            guess_fn.cached_preds = None
        initial_phase = len(set([ch for ch in current_word if ch != "_"])) < 3

        if (current_word == guess_fn.last_word) and (guess_fn.cached_preds is not None):
            sorted_predictions = guess_fn.cached_preds
        else:
            if verbose:
                print(f"already guessed: {guess_fn.already_guessed}")
            predict_output: list[tuple[float, int]] = []
            if len(current_word) <= 5 and small_words_model is not None:
                predict_output = predict(
                    current_word,
                    small_words_model,
                    # already_guessed=guess_fn.already_guessed,
                    verbose=verbose,
                )
            else:
                predict_output = predict(
                    current_word,
                    model,
                    # already_guessed=guess_fn.already_guessed,
                    verbose=verbose,
                    mult_factor=2.5,
                )

            predict_output = [
                (0, c) if c in guess_fn.already_guessed else (p, c)
                for (p, c) in predict_output
            ]
            scrap_output = scrap_g(current_word, hconfig)
            scrap_output = [
                (0, c) if c in guess_fn.already_guessed else (p, c)
                for (p, c) in scrap_output
            ]

            predict_output = sorted(predict_output, reverse=True)
            scrap_output = sorted(scrap_output, reverse=True)

            predict_map = {v: k for k, v in predict_output}
            scrap_map = {v: k for k, v in scrap_output}

            selector_output = []

            if len(scrap_output) > 0:
                if train_phase:
                    if len(scrap_output) > 0 and not initial_phase:
                        x = (
                            [v[0] for v in predict_output[:3]]
                            + [scrap_map[v[1]] for v in predict_output[:3]]
                            + [v[0] for v in scrap_output[:3]]
                            + [predict_map[v[1]] for v in scrap_output[:3]]
                            + [len(current_word)]
                        )

                        replacements = [
                            ord(correct_word[i]) - ord("a")
                            for i in range(len(current_word))
                            if current_word[i] == "_"
                        ]
                        targets = [
                            1 if v in replacements else 0
                            for _, v in (predict_output[:3] + scrap_output[:3])
                        ]
                        append_to_file(x, targets)
                else:
                    selector_model_obj.eval()
                    x = torch.tensor(
                        [v[0] for v in predict_output[:3]]
                        + [scrap_map[v[1]] for v in predict_output[:3]]
                        + [v[0] for v in scrap_output[:3]]
                        + [predict_map[v[1]] for v in scrap_output[:3]]
                        + [len(current_word)],
                        dtype=torch.float32,
                    )
                    targets = [v[1] for v in (predict_output[:3] + scrap_output[:3])]
                    selector_output = [
                        v[1]
                        for v in selector_model(x.unsqueeze(0), targets)[
                            : hconfig.selector_prefix_len
                        ]
                    ]

            sorted_predictions = selector_output + [
                v[1]
                for v in sorted(
                    predict_output[:1] + scrap_output + predict_output[1:], reverse=True
                )
            ]

            sorted_predictions = check_vowels(
                current_word,
                guess_fn.already_guessed,
                rare_triads_set,
                sorted_predictions,
            )
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
    hconfig: HConfig,
    model_object=None,
    small_words_model_object=None,
    test_words_file: str = "w_test.txt",
    start_idx_word: int = 0,
    max_test_words: int | None = None,
    small_words: bool = False,
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

    if small_words:
        test_words = [w for w in test_words if len(w) <= 5]

    if max_test_words and len(test_words) > max_test_words:
        test_words = test_words[start_idx_word : start_idx_word + max_test_words]

    if verbose:
        print(f"Testing on {len(test_words)} words...")

    if model_object is not None:
        model = model_object
    else:
        raise ValueError("Either model_filepath or model_object must be provided.")
    model_results: list[int] = []

    for i, word in tqdm(enumerate(test_words)):
        result = model_single_game(
            hconfig,
            test_word=word,
            model=model,
            small_words_model=small_words_model_object,
            verbose=False,
        )
        model_results.append(result)
        if verbose:
            if (i + 1) % 300 == 0:
                print(
                    f"Played {i + 1}/{len(test_words)} games, winrate: {np.mean(model_results)}"
                )

    model_win_rate = float(np.mean(model_results))
    if verbose:
        print(f"Model win rate: {model_win_rate:.2%}")

    return model_win_rate


def model_single_game(
    hconfig: HConfig,
    model_filepath: str = "models/cb.pth",
    model=None,
    small_words_model=None,
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
        hconfig,
        small_words_model=small_words_model,
        verbose=verbose,
    )
    reset_guesser_state(model_guesser)
    model_result = play_game(test_word, model_guesser, verbose=verbose)
    if model_result == 0:
        words_lost.append(test_word)

    # if model_result == 0 and test_word not in pass_words:
    #     pass_words.append(test_word)
    # print(f"Failed on word: {test_word}. Total failed words: {len(words_lost)}") play = input("play?")
    # if play == "j":
    #     model_guesser = create_model_guesser(
    #         model,
    #         hconfig,
    #         small_words_model=small_words_model,
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
    hconfig = HConfig(
        selector_prefix_len=1, min_non_blanks=5, max_blanks=3, span_start=6
    )

    model_filepath = "models/nn2b.pth_checkpoint_58"
    model.load_state_dict(
        torch.load(model_filepath, weights_only=True, map_location=torch.device(device))
    )
    model.eval()

    # res = model_single_game(hconfig, model=model, test_word="spissitudj", verbose=True)
    # print(res)
    # res1 = []
    # res2 = []
    #
    # for _ in range(5):
    #     s_res = test_model_on_game_play(
    #         hconfig,
    #         model_object=model,
    #         small_words_model_object=None,
    #         verbose=True,
    #         max_test_words=30,
    #     )
    #     e_res = test_model_on_game_play(
    #         hconfig,
    #         model_object=model,
    #         small_words_model_object=small_words_model,
    #         verbose=True,
    #         max_test_words=30,
    #         start_idx_word=400,
    #     )
    #     res1.append(s_res)
    #     res2.append(e_res)
    # print(res1)
    # print(res2)
    s_res = test_model_on_game_play(
        hconfig,
        model_object=model,
        small_words_model_object=None,
        verbose=True,
    )
