import json
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt

from data_load import load_data
from preprocess import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

DATA_PATH = "PetImages"
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = 3000   # None = use all images (very slow)


def main():

    print("--" * 30)
    print("SVM Image Recognition: Cat vs Dog")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] preprocess images...")

    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Step 4: Train SVM
    print("\n[Step 4] Training SVM...")

    model, scaler = train_svm(X_train, y_train)

    joblib.dump(model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    print("SVM training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_svm(model, scaler, X_test)

     # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")

        # Step 7: Save prediction samples
    print("\n[Step 7] Saving prediction samples...")

    num_samples = min(10, len(X_test))

    plt.figure(figsize=(15, 6))

    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)

        image = X_test[i].reshape(IMG_SIZE, IMG_SIZE)

        plt.imshow(image, cmap="gray")

        true_label = classes[y_test[i]]
        pred_label = classes[predictions[i]]

        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/prediction_sample.png",
        dpi=150
    )

    plt.close()

    print(f"Saved: {OUTPUT_DIR}/prediction_sample.png")


if __name__ == "__main__":
    main()