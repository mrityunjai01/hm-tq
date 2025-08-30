import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    f1_score,
    precision_score,
    recall_score,
)
from skops.io import load
from cb.train import train
from prepare.create_dataset import load_dataset
from prepare.data import gen_x_y_for_word


def test_model_on_test_words(
    model_filepath: str = "models/cb.pth",
    test_words_file: str = "w_test.txt",
) -> None:
    """
    Test the trained CatBoost classifier on all test words with multilabel metrics.

    Args:
        model_filepath: Path to the saved model
        test_words_file: Path to test words file
    """
    # Load test words
    if not os.path.exists(test_words_file):
        print(f"Test words file {test_words_file} not found.")
        return

    with open(test_words_file, "r") as f:
        test_words = [w.strip() for w in f.readlines()]

    print(f"Loading {len(test_words)} test words...")

    # Load model
    if not os.path.exists(model_filepath):
        print(f"Model not found at {model_filepath}. Please train first.")
        return

    print(f"Loading model from {model_filepath}...")
    model = load(model_filepath, trusted=["catboost.core.CatBoostClassifier"])

    # Generate test data from all words
    print("Generating test data...")
    all_X = []
    all_Y = []

    for i, word in enumerate(test_words):
        try:
            X_word, Y_word = gen_x_y_for_word(word)
            if len(X_word) > 0:
                all_X.append(X_word)
                all_Y.append(Y_word)

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(test_words)} words")

        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            continue

    if not all_X:
        print("No test data generated!")
        return

    # Combine all test data
    X_test = np.vstack(all_X)
    Y_test = np.vstack(all_Y)

    print(f"Test data shape: X={X_test.shape}, Y={Y_test.shape}")

    # Make predictions
    print("Making predictions...")
    Y_pred = model.predict(X_test)

    # Calculate multilabel classification metrics
    print("\n=== Multilabel Classification Metrics ===")

    # Exact match ratio (all labels must be correct)
    exact_match = accuracy_score(Y_test, Y_pred)
    print(f"Exact match accuracy: {exact_match:.4f}")

    # Hamming loss (average per-label error rate)
    hamming = hamming_loss(Y_test, Y_pred)
    print(f"Hamming loss: {hamming:.4f}")

    # F1 scores
    f1_micro = f1_score(Y_test, Y_pred, average="micro")
    f1_macro = f1_score(Y_test, Y_pred, average="macro")
    f1_weighted = f1_score(Y_test, Y_pred, average="weighted")

    print(f"F1-score (micro): {f1_micro:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")

    # Precision and Recall
    precision_micro = precision_score(Y_test, Y_pred, average="micro")
    precision_macro = precision_score(Y_test, Y_pred, average="macro")

    recall_micro = recall_score(Y_test, Y_pred, average="micro")
    recall_macro = recall_score(Y_test, Y_pred, average="macro")

    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")

    # Per-label metrics
    print("\n=== Per-Label Statistics ===")
    per_label_f1 = f1_score(Y_test, Y_pred, average=None)
    per_label_precision = precision_score(Y_test, Y_pred, average=None)
    per_label_recall = recall_score(Y_test, Y_pred, average=None)

    print(f"Mean per-label F1: {np.mean(per_label_f1):.4f}")
    print(f"Mean per-label precision: {np.mean(per_label_precision):.4f}")
    print(f"Mean per-label recall: {np.mean(per_label_recall):.4f}")

    # Show worst performing labels
    worst_labels = np.argsort(per_label_f1)[:5]
    print("\nWorst performing labels (F1):")
    for idx in worst_labels:
        letter = chr(ord("a") + idx)
        print(
            f"  {letter}: F1={per_label_f1[idx]:.4f}, P={per_label_precision[idx]:.4f}, R={per_label_recall[idx]:.4f}"
        )

    # Show best performing labels
    best_labels = np.argsort(per_label_f1)[-5:][::-1]
    print("\nBest performing labels (F1):")
    for idx in best_labels:
        letter = chr(ord("a") + idx)
        print(
            f"  {letter}: F1={per_label_f1[idx]:.4f}, P={per_label_precision[idx]:.4f}, R={per_label_recall[idx]:.4f}"
        )


if __name__ == "__main__":
    test_model_on_test_words()

