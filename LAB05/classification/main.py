from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data_loader import load_dataset
from preprocessing import standardize_data
from split_data import split_dataset
from svm_model import create_models, train_models
from evaluate import evaluate_models


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    print("=" * 50)
    print("LAB 05 - Support Vector Machine")
    print("Wine Dataset")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------

    X, y, class_names = load_dataset()

    print("\nDataset Information")
    print("-" * 50)
    print(f"Number of samples : {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes : {len(class_names)}")

    print("\nClasses:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    print("\nFirst 5 rows:")
    print(X.head())

    # --------------------------------------------------
    # 2. Split Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\nData Split")
    print("-" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # --------------------------------------------------
    # 3. Standardization
    # --------------------------------------------------

    X_train_scaled, X_test_scaled, scaler = standardize_data(
        X_train,
        X_test
    )

    print("\nStandardization completed.")

    # Save processed data
    np.save(OUTPUT_DIR / "X_train.npy", X_train_scaled)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.pkl"
    )

    # --------------------------------------------------
    # 4. Create SVM Models
    # --------------------------------------------------

    models = create_models()

    print("\nSVM Models")
    print("-" * 50)

    for name in models:
        print(f"- {name}")

    # --------------------------------------------------
    # 5. Train Models
    # --------------------------------------------------

    trained_models = train_models(
        models,
        X_train_scaled,
        y_train
    )

    print("\nTraining completed.")

    # --------------------------------------------------
    # 6. Evaluate Models
    # --------------------------------------------------

    print("\nModel Accuracy")
    print("-" * 50)

    results = evaluate_models(
        trained_models,
        X_test_scaled,
        y_test,
        class_names,
        OUTPUT_DIR
    )

    results_df = pd.DataFrame(results)

    print("\n")
    print(results_df.to_string(index=False))

    # Save accuracy results
    results_df.to_csv(
        OUTPUT_DIR / "accuracy_results.csv",
        index=False
    )

    # --------------------------------------------------
    # 7. Save Models
    # --------------------------------------------------

    joblib.dump(
        trained_models,
        OUTPUT_DIR / "svm_models.pkl"
    )

    # --------------------------------------------------
    # 8. Prediction Example
    # --------------------------------------------------

    print("\nPrediction Example")
    print("-" * 50)

    sample_data = X_test_scaled[:5]

    for name, model in trained_models.items():

        predictions = model.predict(sample_data)

        print(f"\n{name} Kernel:")

        for i, prediction in enumerate(predictions):
            print(
                f"Sample {i + 1}: "
                f"Class {prediction} "
                f"({class_names[prediction]})"
            )

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("LAB 05 completed.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()