import os
import joblib
import numpy as np

from sklearn.metrics import accuracy_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)


def main():

    print("==============================")
    print("Testing Saved SVM Model")
    print("==============================")

    # Load model
    model = joblib.load(
        os.path.join(
            OUTPUT_DIR,
            "svm_model.pkl"
        )
    )

    # Load scaler
    scaler = joblib.load(
        os.path.join(
            OUTPUT_DIR,
            "scaler.pkl"
        )
    )

    # Load test data
    X_test = np.load(
        os.path.join(
            OUTPUT_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            OUTPUT_DIR,
            "y_test.npy"
        )
    )

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nModel loaded successfully")

    print(f"Test samples: {len(X_test)}")

    print(f"Accuracy: {accuracy:.4f}")

    print("\nSample predictions:")

    for i in range(min(10, len(y_pred))):

        actual = (
            "Good"
            if y_test[i] == 1
            else "Not Good"
        )

        predicted = (
            "Good"
            if y_pred[i] == 1
            else "Not Good"
        )

        print(
            f"{i + 1}. "
            f"Actual: {actual:<8} "
            f"Predicted: {predicted}"
        )


if __name__ == "__main__":
    main()