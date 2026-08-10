import pandas as pd


def load_data(path):
    """
    Load red wine quality dataset.
    """

    try:
        df = pd.read_csv(path, sep=";")

        # If dataset uses comma instead of semicolon
        if len(df.columns) == 1:
            df = pd.read_csv(path)

    except Exception as e:
        raise RuntimeError(f"Cannot load dataset: {e}")

    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df