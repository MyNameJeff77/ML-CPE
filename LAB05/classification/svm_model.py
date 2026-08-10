from sklearn.svm import SVC


def create_models():

    models = {

        "Linear": SVC(
            kernel="linear",
            random_state=42
        ),

        "Polynomial": SVC(
            kernel="poly",
            degree=3,
            random_state=42
        ),

        "RBF": SVC(
            kernel="rbf",
            random_state=42
        )

    }

    return models


def train_models(models, X_train, y_train):

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models