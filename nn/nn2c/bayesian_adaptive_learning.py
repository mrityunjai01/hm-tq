from dataclasses import dataclass
import os
import pickle
import optuna

import numpy as np
import torch

from game.play_game import play_game
from .hconfig import HConfig
from .runner import test_model_on_game_play
from .selector import SelectorNN, append_to_file
from .scrap import scrap_g
from tqdm import tqdm

from .model1 import HangmanNet
from .predict import predict


from nn.nn2b.model1 import HangmanNet

stage_hyperparam = False


def inference_objective(trial):
    hconfig = HConfig(
        selector_prefix_len=1,
        min_non_blanks=trial.suggest_int("min_non_blanks", 4, 10),
        max_blanks=trial.suggest_int("max_blanks", 1, 3),
        mult_factor=trial.suggest_float("mult_factor", 1.0, 4.0, step=0.1),
    )
    device = "cpu"
    model = HangmanNet(vocab_size=27, device=device, num_layers=3).to(device)
    model = torch.compile(model, mode="max-autotune")

    model_filepath = "models/nn2b.pth_checkpoint_58"
    model.load_state_dict(
        torch.load(model_filepath, weights_only=True, map_location=torch.device(device))
    )
    return test_model_on_game_play(
        model_object=model,
        hconfig=hconfig,
        verbose=True,
        max_test_words=30 if stage_hyperparam else 300,
    )


def run_one(hconfig: HConfig):
    device = "cpu"
    model = HangmanNet(vocab_size=27, device=device, num_layers=3).to(device)
    model = torch.compile(model, mode="max-autotune")

    model_filepath = "models/nn2b.pth_checkpoint_58"
    model.load_state_dict(
        torch.load(model_filepath, weights_only=True, map_location=torch.device(device))
    )
    return test_model_on_game_play(
        model_object=model,
        hconfig=hconfig,
        verbose=True,
        max_test_words=30 if stage_hyperparam else 300,
    )


def bayesian_adaptive_learning():
    study = optuna.create_study(direction="maximize")
    study.optimize(inference_objective, n_trials=2 if stage_hyperparam else 100)
    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)
    with open("models/bayesian_optimization_study.pkl", "wb") as f:
        pickle.dump(study, f)


if __name__ == "__main__":
    # run_one(
    #     HConfig(selector_prefix_len=3, min_non_blanks=5, max_blanks=1, span_start=6)
    # )
    # 'selector_prefix_len': 3, 'min_non_blanks': 5, 'max_blanks': 1, 'span_start': 6}.
    bayesian_adaptive_learning()
