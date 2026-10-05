import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")

MODELS = [
    "vgg16", "vgg19", "inceptionv3",
    "resnet101", "mobilenetv2", "efficientnetb0"
]

DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "inceptionv3": "InceptionV3",
    "resnet101": "ResNet101",
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0"
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_results():
    histories, metrics = {}, {}

    for model in MODELS:
        history_file = OUTPUT_DIR / model / "history.json"
        metrics_file = OUTPUT_DIR / model / "metrics.json"

        if history_file.exists():
            histories[model] = load_json(history_file)
        else:
            print(f"WARNING: ไม่พบ {history_file}")

        if metrics_file.exists():
            metrics[model] = load_json(metrics_file)

    return histories, metrics

def line_plot(histories, key, ylabel, title, filename):
    plt.figure(figsize=(11, 7))

    for model, h in histories.items():
        plt.plot(
            h[key],
            label=DISPLAY_NAMES[model],
            linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=250)
    plt.close()

def make_summary_table(histories, metrics):
    rows = []

    for model in MODELS:
        if model not in histories:
            continue

        h = histories[model]
        best_val_acc = max(h["val_accuracy"])
        best_epoch = int(np.argmax(h["val_accuracy"])) + 1

        test_acc = metrics.get(model, {}).get("test_accuracy")
        test_loss = metrics.get(model, {}).get("test_loss")
        params = metrics.get(model, {}).get("parameters")

        rows.append({
            "Model": DISPLAY_NAMES[model],
            "Best Val Accuracy (%)": round(best_val_acc * 100, 2),
            "Best Epoch": best_epoch,
            "Test Accuracy (%)":
                round(test_acc * 100, 2) if test_acc is not None else None,
            "Test Loss":
                round(test_loss, 4) if test_loss is not None else None,
            "Parameters": params
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== MODEL COMPARISON =====")
    print(df.to_string(index=False))
    return df

def plot_test_accuracy(df):
    df = df.dropna(subset=["Test Accuracy (%)"])

    plt.figure(figsize=(11, 6))
    x = np.arange(len(df))

    plt.bar(x, df["Test Accuracy (%)"])
    plt.xticks(x, df["Model"], rotation=20)
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Model")
    plt.title("Test Accuracy Comparison - 6 DCNN Models")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(df["Test Accuracy (%)"]):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_test_accuracy.png", dpi=250)
    plt.close()

def main():
    histories, metrics = load_results()

    if not histories:
        raise FileNotFoundError(
            "ไม่พบผลการ train ใน outputs/ กรุณารัน train_models.py ก่อน"
        )

    line_plot(
        histories, "accuracy", "Accuracy",
        "Training Accuracy Comparison - 6 DCNN Models",
        "comparison_training_accuracy.png"
    )
    line_plot(
        histories, "val_accuracy", "Validation Accuracy",
        "Validation Accuracy Comparison - 6 DCNN Models",
        "comparison_validation_accuracy.png"
    )
    line_plot(
        histories, "loss", "Loss",
        "Training Loss Comparison - 6 DCNN Models",
        "comparison_training_loss.png"
    )
    line_plot(
        histories, "val_loss", "Validation Loss",
        "Validation Loss Comparison - 6 DCNN Models",
        "comparison_validation_loss.png"
    )

    df = make_summary_table(histories, metrics)

    if len(df) > 0 and "Test Accuracy (%)" in df.columns:
        plot_test_accuracy(df)

    print("\nสร้างกราฟเปรียบเทียบเรียบร้อยแล้ว")
    print("ดูผลทั้งหมดใน outputs/")

if __name__ == "__main__":
    main()
