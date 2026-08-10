from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_models(
    models,
    X_test,
    y_test,
    class_names,
    output_dir
):

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        results.append({
            "Kernel": name,
            "Accuracy": accuracy
        })

        print(
            f"{name} Kernel Accuracy: "
            f"{accuracy:.4f}"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        plt.figure(figsize=(6, 5))

        plt.imshow(cm)

        plt.title(
            f"Confusion Matrix - {name}"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.xticks(
            range(len(class_names)),
            class_names,
            rotation=45
        )

        plt.yticks(
            range(len(class_names)),
            class_names
        )

        for i in range(len(cm)):
            for j in range(len(cm[i])):

                plt.text(
                    j,
                    i,
                    cm[i][j],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        filename = (
            output_dir
            / f"confusion_matrix_{name.lower()}.png"
        )

        plt.savefig(filename)
        plt.close()

    return results