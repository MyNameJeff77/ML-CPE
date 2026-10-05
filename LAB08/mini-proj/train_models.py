import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import importlib
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import make_datasets
from preprocessing import get_preprocessing

EPOCHS = 10
LEARNING_RATE = 1e-4

SPLIT_DIR = Path("split_data")
OUTPUT_DIR = Path("outputs")

MODELS = {
    "vgg16": "models.vgg16_model",
    "vgg19": "models.vgg19_model",
    "inceptionv3": "models.inceptionv3_model",
    "resnet101": "models.resnet101_model",
    "mobilenetv2": "models.mobilenetv2_model",
    "efficientnetb0": "models.efficientnetb0_model",
}

def save_history(history, output_dir):
    data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def plot_history(history, model_name, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=200)
    plt.close()

def train_one(model_name, module_name, train_ds, val_ds, test_ds):
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 70)

    module = importlib.import_module(module_name)
    base_model = module.build_model()
    preprocess = get_preprocessing(model_name)

    # ทำ preprocessing เป็นส่วนหนึ่งของโมเดล
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    x = tf.keras.layers.Lambda(
        lambda img: preprocess(tf.cast(img, tf.float32)),
        name="preprocess"
    )(inputs)
    outputs = base_model(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    save_history(history, output_dir)
    plot_history(history, model_name, output_dir)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "epochs_trained": len(history.history["loss"]),
            "parameters": int(model.count_params())
        }, f, indent=2)

    model.save(output_dir / f"{model_name}.keras")
    print(f"{model_name} Test Accuracy = {test_accuracy * 100:.2f}%")

def main():
    train_dir = SPLIT_DIR / "train"
    val_dir = SPLIT_DIR / "val"
    test_dir = SPLIT_DIR / "test"

    if not train_dir.exists():
        raise FileNotFoundError("ยังไม่มี split_data/ กรุณารัน split_data.py ก่อน")

    train_ds, val_ds, test_ds, class_names = make_datasets(
        train_dir, val_dir, test_dir
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    for model_name, module_name in MODELS.items():
        train_one(model_name, module_name, train_ds, val_ds, test_ds)

    print("\nทดลองครบทั้ง 6 โมเดลแล้ว")
    print("ต่อไปให้รัน compare_models.py")

if __name__ == "__main__":
    main()
