from sklearn.svm import SVC


def train_svm(X_train, y_train):
    """
    Train SVM classifier.
    """

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nSVM training completed")

    return model