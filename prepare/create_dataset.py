import os
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from pathlib import Path
from prepare.data import gen_x_y_for_word
from prepare.separate_train_test import separate_train_test

from stage import STAGE


def create_dataset(
    words_file: str = "w_train.txt",
    output_dir: str = "data",
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


if __name__ == "__main__":
    # Create dataset from training words
    create_dataset(
        words_file="w_train.txt",
        output_dir="data",
    )
