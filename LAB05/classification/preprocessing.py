import pandas as pd


def preprocess_data(df, quality_threshold=7):
    """
    Prepare features and target.

    quality >= 7 -> Good Wine (1)
    quality < 7  -> Not Good Wine (0)
    """

    df = df.copy()

    # Create binary target
    df["quality_label"] = (
        df["quality"] >= quality_threshold
    ).astype(int)

    # Features
    X = df.drop(columns=["quality", "quality_label"])

    # Target
    y = df["quality_label"]

    print("\nPreprocessing completed")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    return X, y