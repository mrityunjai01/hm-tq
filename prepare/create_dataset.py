import os
import pandas as pd
import numpy as np
from pathlib import Path
from prepare.data import gen_x_y_for_word
from prepare.separate_train_test import separate_train_test


def create_dataset(
    words_file: str = "w_train.txt",
    output_dir: str = "data",
    max_words: int = None,
) -> None:
    """
    Generate training data from words and save as single parquet files.

    Args:
        words_file: Path to the words file
        output_dir: Directory to save parquet files
        max_words: Maximum number of words to process (None for all)
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)

    # Load words
    if not os.path.exists(words_file):
        print(f"Words file {words_file} not found. Creating train/test split...")
        separate_train_test()

    with open(words_file, "r") as f:
        words = [w.strip() for w in f.readlines()]

    if max_words:
        words = words[:max_words]

    print(f"Processing {len(words)} words...")

    all_x_data = []
    all_y_data = []

    for i, word in enumerate(words):
        try:
            x_word, y_word = gen_x_y_for_word(word)

            if len(x_word) > 0:  # Only add if data was generated
                all_x_data.append(x_word)
                all_y_data.append(y_word)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(words)} words")

        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            continue

    # Concatenate all data
    X = np.vstack(all_x_data)
    Y = np.vstack(all_y_data)

    # Create DataFrames and save
    x_columns = [f"x_{i}" for i in range(X.shape[1])]
    y_columns = [f"y_{i}" for i in range(Y.shape[1])]

    x_df = pd.DataFrame(X, columns=x_columns)
    y_df = pd.DataFrame(Y, columns=y_columns)

    x_path = os.path.join(output_dir, "X_train.parquet")
    y_path = os.path.join(output_dir, "Y_train.parquet")

    x_df.to_parquet(x_path, compression="snappy")
    y_df.to_parquet(y_path, compression="snappy")

    print(f"Dataset saved: {len(X)} samples")
    print(f"X shape: {X.shape} -> {x_path}")
    print(f"Y shape: {Y.shape} -> {y_path}")


def load_dataset(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from parquet files.

    Args:
        data_dir: Directory containing parquet files

    Returns:
        Tuple of (X, Y) numpy arrays
    """
    x_path = os.path.join(data_dir, "X_train.parquet")
    y_path = os.path.join(data_dir, "Y_train.parquet")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Dataset files not found in {data_dir}")

    x_df = pd.read_parquet(x_path)
    y_df = pd.read_parquet(y_path)

    X = x_df.values.astype(np.float32)
    Y = y_df.values.astype(np.float32)

    print(f"Loaded dataset: X shape {X.shape}, Y shape {Y.shape}")

    return X, Y


if __name__ == "__main__":
    # Create dataset from training words
    create_dataset(
        words_file="w_train.txt",
        output_dir="data",
    )

    # Example of loading the dataset
    try:
        X, Y = load_dataset("data")
        print(f"Successfully loaded dataset with shapes: X={X.shape}, Y={Y.shape}")
    except FileNotFoundError as e:
        print(f"Could not load dataset: {e}")
