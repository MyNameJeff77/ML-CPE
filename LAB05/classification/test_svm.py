from pathlib import Path
import sys

import numpy as np

sys.path.append(
    str(Path(__file__).parent)
)

from data_loader import load_dataset
from preprocessing import standardize_data
from split_data import split_dataset
from svm_model import create_models, train_models


def test_dataset():

    X, y, class_names = load_dataset()

    assert X.shape[0] == 178
    assert X.shape[1] == 13
    assert len(class_names) == 3

    print("test_dataset: PASS")


def test_split():

    X, y, _ = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    assert len(X_train) == 142
    assert len(X_test) == 36

    print("test_split: PASS")


def test_svm_models():

    X, y, _ = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    X_train_scaled, X_test_scaled, scaler = standardize_data(
        X_train,
        X_test
    )

    models = create_models()

    trained_models = train_models(
        models,
        X_train_scaled,
        y_train
    )

    assert len(trained_models) == 3

    for name, model in trained_models.items():

        predictions = model.predict(
            X_test_scaled
        )

        assert len(predictions) == len(y_test)

    print("test_svm_models: PASS")


if __name__ == "__main__":

    test_dataset()
    test_split()
    test_svm_models()

    print("\nAll tests passed.")