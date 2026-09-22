import numpy as np

from sklearn.model_selection import train_test_split


def split_dataset(
    X,
    y,
    test_size=0.10,
    val_size=0.10,
    random_state=42
):
    """
    Split dataset into:

    Train      = 80%
    Validation = 10%
    Test       = 10%

    Stratified split keeps Cat/Dog proportions balanced.
    """

    # ---------------------------------------------------------
    # STEP 1
    # Split 10% for TEST
    # ---------------------------------------------------------

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True
    )

    # ---------------------------------------------------------
    # STEP 2
    # Split VALIDATION from remaining data
    #
    # Remaining = 90%
    # Need validation = 10% of total
    #
    # 0.10 / 0.90 = 0.1111
    # ---------------------------------------------------------

    val_ratio = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
        shuffle=True
    )

    # ---------------------------------------------------------
    # Shuffle training data
    # ---------------------------------------------------------

    train_indices = np.random.default_rng(
        random_state
    ).permutation(len(X_train))

    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    # ---------------------------------------------------------
    # Print distribution
    # ---------------------------------------------------------

    print("\nDataset Split")
    print("=" * 50)

    print(f"Training   : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Testing    : {len(X_test)}")

    print("\nClass Distribution")
    print("-" * 50)

    print(
        f"Train      Cat: {np.sum(y_train == 0)} | "
        f"Dog: {np.sum(y_train == 1)}"
    )

    print(
        f"Validation Cat: {np.sum(y_val == 0)} | "
        f"Dog: {np.sum(y_val == 1)}"
    )

    print(
        f"Test       Cat: {np.sum(y_test == 0)} | "
        f"Dog: {np.sum(y_test == 1)}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )