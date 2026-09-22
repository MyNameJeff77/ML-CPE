import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset

from cnn_model import (
    train_model,
    predict_model
)

from evaluate import (
    evaluate_model,
    plot_history
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "PetImages"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

# ---------------------------------------------------------
# Image Size
# ---------------------------------------------------------

IMG_SIZE = 100

# ---------------------------------------------------------
# Dataset Split
# ---------------------------------------------------------

TEST_SIZE = 0.10

VAL_SIZE = 0.10

# ---------------------------------------------------------
# Maximum Images
#
# 3000 Cat
# 3000 Dog
# = 6000 images
# ---------------------------------------------------------

MAX_PER_CLASS = 3000

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

EPOCHS = 50

BATCH_SIZE = 32


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)

    print(
        "CNN Image Recognition: Cat vs Dog"
    )

    print("=" * 60)

    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # =====================================================
    # STEP 1
    # LOAD DATASET
    # =====================================================

    print("\n")
    print("=" * 60)
    print("[Step 1] Loading dataset...")
    print("=" * 60)

    images, labels, classes = load_data(

        DATA_PATH,

        IMG_SIZE,

        MAX_PER_CLASS
    )

    # -----------------------------------------------------
    # SAVE LABELS
    # -----------------------------------------------------

    np.save(

        os.path.join(
            OUTPUT_DIR,
            "labels.npy"
        ),

        labels
    )

    # -----------------------------------------------------
    # SAVE CLASS NAMES
    # -----------------------------------------------------

    with open(

        os.path.join(
            OUTPUT_DIR,
            "classes.json"
        ),

        "w"
    ) as f:

        json.dump(
            classes,
            f,
            indent=4
        )

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    print(
        "\nDataset loaded successfully."
    )

    print(
        f"Total images : {len(images)}"
    )

    print(
        f"Classes      : {classes}"
    )

    # -----------------------------------------------------
    # CLASS DISTRIBUTION
    # -----------------------------------------------------

    for class_id, class_name in enumerate(classes):

        count = np.sum(
            labels == class_id
        )

        print(
            f"{class_name:10s}: "
            f"{count} images"
        )

    # =====================================================
    # STEP 2
    # PREPROCESSING
    # =====================================================

    print("\n")
    print("=" * 60)
    print("[Step 2] Preprocessing images...")
    print("=" * 60)

    X = to_features(
        images
    )

    y = labels

    print(
        f"Feature shape: {X.shape}"
    )

    # -----------------------------------------------------
    # SAVE FEATURES
    # -----------------------------------------------------

    np.save(

        os.path.join(
            OUTPUT_DIR,
            "features.npy"
        ),

        X
    )

    # =====================================================
    # STEP 3
    # SPLIT DATASET
    # =====================================================

    print("\n")
    print("=" * 60)
    print("[Step 3] Splitting dataset...")
    print("=" * 60)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test

    ) = split_dataset(

        X,

        y,

        TEST_SIZE,

        VAL_SIZE
    )

    # =====================================================
    # SAVE SPLIT DATA
    # =====================================================

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_val.npy"
        ),
        X_val
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_val.npy"
        ),
        y_val
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_test.npy"
        ),
        y_test
    )

    # =====================================================
    # STEP 4
    # TRAIN MODEL
    # =====================================================

    print("\n")
    print("=" * 60)
    print("[Step 4] Training model...")
    print("=" * 60)

    model, history = train_model(

        X_train,

        y_train,

        X_val,

        y_val,

        len(classes),

        OUTPUT_DIR,

        EPOCHS,

        BATCH_SIZE
    )

    print(
        "\nTraining completed."
    )

    # =====================================================
    # STEP 5
    # TEST MODEL
    # =====================================================

    print("\n")
    print("=" * 60)
    print("[Step 5] Testing model...")
    print("=" * 60)

    predictions = predict_model(

        model,

        X_test
    )

    # =====================================================
    # STEP 6
    # EVALUATE
    # =====================================================

    print("\n")
    print("=" * 60)
    print("[Step 6] Evaluating model...")
    print("=" * 60)

    evaluate_model(

        y_test,

        predictions,

        classes,

        save_path=os.path.join(

            OUTPUT_DIR,

            "confusion_matrix.png"
        )
    )

    # =====================================================
    # STEP 7
    # TRAINING GRAPH
    # =====================================================

    print("\n")
    print("=" * 60)
    print("[Step 7] Creating training graph...")
    print("=" * 60)

    plot_history(

        history,

        os.path.join(

            OUTPUT_DIR,

            "training_history.png"
        )
    )

    # =====================================================
    # DONE
    # =====================================================

    print("\n")

    print("=" * 60)

    print(
        "PROCESS COMPLETED"
    )

    print("=" * 60)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()