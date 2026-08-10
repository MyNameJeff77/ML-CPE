import os
import joblib
import numpy as np

from data_loader import load_data
from preprocessing import preprocess_data
from split_data import split_data
from svm_model import train_svm
from evaluate import evaluate_model

from sklearn.preprocessing import StandardScaler


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "dataset.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)


# =========================
# Main
# =========================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("==============================")
    print("LAB05 - SVM Wine Classification")
    print("==============================")

    # 1. Load dataset
    df = load_data(DATA_PATH)

    # 2. Preprocessing
    X, y = preprocess_data(
        df,
        quality_threshold=7
    )

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Convert to numpy
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # 4. Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature scaling completed")

    # Save scaler
    scaler_path = os.path.join(
        OUTPUT_DIR,
        "scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    # 5. Save train/test data
    np.save(
        os.path.join(OUTPUT_DIR, "X_train.npy"),
        X_train_scaled
    )

    np.save(
        os.path.join(OUTPUT_DIR, "X_test.npy"),
        X_test_scaled
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_train.npy"),
        y_train
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_test.npy"),
        y_test
    )

    # 6. Train SVM
    model = train_svm(
        X_train_scaled,
        y_train
    )

    # 7. Save model
    model_path = os.path.join(
        OUTPUT_DIR,
        "svm_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"\nModel saved to:")
    print(model_path)

    # 8. Evaluate
    accuracy = evaluate_model(
        model,
        X_test_scaled,
        y_test,
        OUTPUT_DIR
    )

    print("\n==============================")
    print("Training completed")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()