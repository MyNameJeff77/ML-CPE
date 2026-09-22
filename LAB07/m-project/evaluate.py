import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    y_test,
    predictions,
    classes,
    save_path=None
):

    labels = list(
        range(len(classes))
    )

    # =====================================================
    # ACCURACY
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "\n------------ Evaluation ------------------"
    )

    print(
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print(
        "\nClassification Report:"
    )

    report = classification_report(

        y_test,

        predictions,

        labels=labels,

        target_names=classes,

        zero_division=0
    )

    print(
        report
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    matrix = confusion_matrix(

        y_test,

        predictions,

        labels=labels
    )

    print(
        "Confusion Matrix:"
    )

    print(
        matrix
    )

    # =====================================================
    # SAVE CONFUSION MATRIX
    # =====================================================

    if save_path:

        plot_confusion_matrix(

            matrix,

            classes,

            save_path
        )

        print(
            f"Saved: {save_path}"
        )

    return accuracy


# =========================================================
# CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(
    matrix,
    classes,
    save_path
):

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_xticklabels(
        classes
    )

    ax.set_yticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_yticklabels(
        classes
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    threshold = matrix.max() / 2

    for i in range(
        len(classes)
    ):

        for j in range(
            len(classes)
        ):

            ax.text(

                j,

                i,

                matrix[i, j],

                ha="center",

                va="center",

                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                )
            )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(
        fig
    )


# =========================================================
# TRAINING HISTORY
# =========================================================

def plot_history(
    history,
    save_path
):

    train_accuracy = history.history[
        "accuracy"
    ]

    val_accuracy = history.history[
        "val_accuracy"
    ]

    train_loss = history.history[
        "loss"
    ]

    val_loss = history.history[
        "val_loss"
    ]

    epochs = range(
        1,
        len(train_accuracy) + 1
    )

    # =====================================================
    # BEST VALUES
    # =====================================================

    best_val_accuracy = max(
        val_accuracy
    )

    best_epoch = (
        val_accuracy.index(
            best_val_accuracy
        ) + 1
    )

    print(
        f"\nBest Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    # =====================================================
    # FIGURE
    # =====================================================

    fig, axes = plt.subplots(

        1,

        2,

        figsize=(13, 5)
    )

    # =====================================================
    # ACCURACY
    # =====================================================

    axes[0].plot(

        epochs,

        train_accuracy,

        linewidth=2,

        label="train"
    )

    axes[0].plot(

        epochs,

        val_accuracy,

        linewidth=2,

        label="validation"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Accuracy"
    )

    axes[0].set_title(
        "Training and Validation Accuracy"
    )

    axes[0].set_xlim(

        1,

        len(train_accuracy)
    )

    axes[0].grid(
        True,
        alpha=0.25
    )

    axes[0].legend()

    # =====================================================
    # LOSS
    # =====================================================

    axes[1].plot(

        epochs,

        train_loss,

        linewidth=2,

        label="train"
    )

    axes[1].plot(

        epochs,

        val_loss,

        linewidth=2,

        label="validation"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Loss"
    )

    axes[1].set_title(
        "Training and Validation Loss"
    )

    axes[1].set_xlim(

        1,

        len(train_loss)
    )

    axes[1].grid(
        True,
        alpha=0.25
    )

    axes[1].legend()

    # =====================================================
    # SAVE
    # =====================================================

    fig.tight_layout()

    fig.savefig(

        save_path,

        dpi=150
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {save_path}"
    )