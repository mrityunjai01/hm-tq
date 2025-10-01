#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for CatBoost hangman model.
"""

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier
from prepare.create_dataset import load_dataset


def objective(trial):
    """
    Optuna objective function to minimize.

    Args:
        trial: Optuna trial object

    Returns:
        Mean cross-validation score (negated for minimization)
    """
    # Load dataset
    X, Y = load_dataset("data")

    # Sample hyperparameters
    iterations = trial.suggest_int("iterations", 500, 2000, step=500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    depth = trial.suggest_int("depth", 4, 10)

    # Create model
    cb = MultiOutputClassifier(
        CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=False,
            random_seed=42,
        )
    )

    # Use cross-validation with hamming loss (lower is better)
    scores = cross_val_score(
        cb,
        X,
        Y,
        cv=3,  # 3-fold CV for speed
        scoring="neg_hamming_loss",  # Minimize hamming loss
        n_jobs=-1,
    )

    # Return mean score (already negative for minimization)
    mean_score = np.mean(scores)

    # Report intermediate result for pruning
    trial.report(mean_score, 0)

    return mean_score


def main():
    """Run hyperparameter tuning."""
    print("=== Optuna Hyperparameter Tuning ===")

    # Check if dataset exists
    try:
        X, Y = load_dataset("data")
        print(f"Dataset loaded: X={X.shape}, Y={Y.shape}")
    except FileNotFoundError:
        print("Dataset not found. Please run 'python prepare/create_dataset.py' first.")
        return

    # Create study
    study = optuna.create_study(
        direction="maximize",  # Maximize neg_hamming_loss (minimize hamming_loss)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    # Optimize
    n_trials = 50
    print(f"Starting optimization with {n_trials} trials...")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    print("\n=== Optimization Results ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score (neg_hamming_loss): {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_trial.params}")

    # Show parameter importance
    print("\n=== Parameter Importance ===")
    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"{param}: {imp:.4f}")

    # Train final model with best parameters
    print("\n=== Training Final Model ===")
    best_params = study.best_trial.params

    cb_final = MultiOutputClassifier(
        CatBoostClassifier(
            iterations=best_params["iterations"],
            learning_rate=best_params["learning_rate"],
            depth=best_params["depth"],
            verbose=True,
            random_seed=42,
        )
    )

    cb_final.fit(X, Y)

    # Save best model
    import os
    from skops.io import dump

    os.makedirs("models", exist_ok=True)
    model_path = "models/cb_tuned.pth"
    dump(cb_final, model_path)

    print(f"\nBest model saved to: {model_path}")
    print("To test the tuned model, run: python cb/test.py")


if __name__ == "__main__":
    main()

