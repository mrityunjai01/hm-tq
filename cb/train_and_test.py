#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for CatBoost hangman model.
"""

import os

from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from skops.io import dump

from cb.cb_game_test import test_model_on_game_play
from prepare.create_dataset import load_dataset

from stage import STAGE, GPU


def main():
    """A single train and test"""

    iterations = 2 if STAGE else 300
    learning_rate = 0.04
    depth = 2 if STAGE else 5

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

    test_score = test_model_on_game_play(
        model_filepath=None,
        model_object=cb_final,
        max_test_words=100 if STAGE else None,
    )
    print(f"Final test score: {test_score}")
    model_filepath = "models/cb_tuned.pth"
    dump(cb_final, model_filepath)


if __name__ == "__main__":
    main()
