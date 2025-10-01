from catboost import CatBoostClassifier
import numpy as np
from numpy.typing import NDArray
from sklearn.multioutput import MultiOutputClassifier
from skops.io import dump


def train(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    model_filepath: str = "models/cb.pth",
    **kwargs,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
) -> None:
    cb = MultiOutputClassifier(
        CatBoostClassifier(
            iterations=20,
            learning_rate=0.1,
            depth=8,
            random_seed=42,
            cat_features=list(range(26)),
            **kwargs,
        )
    )
    cb: MultiOutputClassifier = cb.fit(x, y)  # pyright: ignore[reportUnknownMemberType]
    dump(cb, model_filepath)
