from sklearn.datasets import load_wine
import pandas as pd


def load_dataset():
    wine = load_wine()

    X = pd.DataFrame(
        wine.data,
        columns=wine.feature_names
    )

    y = pd.Series(
        wine.target,
        name="target"
    )

    return X, y, wine.target_names