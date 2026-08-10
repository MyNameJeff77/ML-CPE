import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test, output_path):
    """
    Evaluate SVM model.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("SVM Evaluation")
    print("==============================")

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Not Good", "Good"]
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    os.makedirs(output_path, exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Good", "Good"]
    )

    disp.plot()

    plt.title("SVM - Wine Quality Confusion Matrix")
    plt.tight_layout()

    save_path = os.path.join(
        output_path,
        "confusion_matrix.png"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"\nConfusion matrix saved to:")
    print(save_path)

    return accuracy