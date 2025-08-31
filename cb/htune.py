#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for CatBoost hangman model.
"""

import os

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from skops.io import dump

from cb.cb_game_test import test_model_on_game_play
from prepare.create_dataset import load_dataset

from stage import STAGE, GPU


def objective(trial):
    """
    Optuna objective function to minimize.

    Args:
        trial: Optuna trial object

    Returns:
        Mean cross-validation score (negated for minimization)
    """
    X, Y = load_dataset("data")
    if STAGE:
        X = X[:2000]
        Y = Y[:2000]

    if STAGE:
        iterations = trial.suggest_int("iterations", 2, 9, step=1)
    else:
        iterations = trial.suggest_int("iterations", 200, 900, step=100)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    depth = trial.suggest_int("depth", 4, 10)

    cb = MultiOutputClassifier(
        CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=42,
            cat_features=list(range(26)),
            task_type="GPU" if GPU else "CPU",
        )
    )
    cb: MultiOutputClassifier = cb.fit(
        X,
        Y,
    )  # pyright: ignore[reportUnknownMemberType]
    mean_score = test_model_on_game_play(
        model_filepath=None, model_object=cb, max_test_words=100 if STAGE else None
    )

    return mean_score


def main():
    """Run hyperparameter tuning."""

    study = optuna.create_study(
        direction="maximize",  # Maximize neg_hamming_loss (minimize hamming_loss)
    )

    n_trials = 3 if STAGE else 50
    print(f"Starting optimization with {n_trials} trials...")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score (neg_hamming_loss): {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_trial.params}")

    print("\n=== Parameter Importance ===")
    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"{param}: {imp:.4f}")

    print("\n=== Training Final Model ===")
    best_params = study.best_trial.params
    iterations, learning_rate, depth = (
        best_params["iterations"],
        best_params["learning_rate"],
        best_params["depth"],
    )

    cb_final = MultiOutputClassifier(
        CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=42,
            cat_features=list(range(26)),
            task_type="GPU" if GPU else "CPU",
        )
    )

    X, Y = load_dataset("data")
    if STAGE:
        X = X[:2000]
        Y = Y[:2000]

    cb_final: MultiOutputClassifier = cb_final.fit(
        X,
        Y,
    )  # pyright: ignore[reportUnknownMemberType]
    model_filepath = "models/cb_tuned.pth"
    dump(cb_final, model_filepath)


if __name__ == "__main__":
    main()
