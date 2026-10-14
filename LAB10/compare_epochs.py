from pathlib import Path
import csv
import time
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "epoch_comparison"

MODEL_NAME = "yolov8n.pt"
EPOCHS_LIST = [10, 20, 30]
IMAGE_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0
SEED = 42


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Check PyTorch GPU installation."
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("       YOLOv8 AQUARIUM EPOCH COMPARISON")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {EPOCHS_LIST}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    results_table = []

    for epochs in EPOCHS_LIST:
        run_name = f"aquarium_{epochs}epochs"

        print("\n" + "=" * 60)
        print(f"TRAINING: {epochs} EPOCHS")
        print("=" * 60)

        model = YOLO(MODEL_NAME)
        start_time = time.time()

        model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=0,
            amp=True,
            cache=False,
            patience=epochs + 1,
            project=str(OUTPUT_DIR),
            name=run_name,
            exist_ok=False,
            save=True,
            plots=True,
            seed=SEED,
            deterministic=True,
            verbose=True
        )

        train_seconds = time.time() - start_time
        save_dir = Path(model.trainer.save_dir)
        best_path = save_dir / "weights" / "best.pt"

        if not best_path.exists():
            raise FileNotFoundError(
                f"Best model not found: {best_path}"
            )

        # Evaluate each model on validation data.
        # Keep test data untouched until model selection.
        print(f"\nEvaluating {epochs}-epoch model on validation set...")

        best_model = YOLO(str(best_path))
        metrics = best_model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=0,
            plots=True,
            project=str(OUTPUT_DIR),
            name=f"{run_name}_validation",
            exist_ok=False,
            verbose=False
        )

        row = {
            "epochs_requested": epochs,
            "epochs_completed": int(model.trainer.epoch + 1),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "training_minutes": train_seconds / 60,
            "best_model": str(best_path)
        }

        results_table.append(row)

        print("\nValidation result:")
        print(f"Precision: {row['precision']:.4f}")
        print(f"Recall:    {row['recall']:.4f}")
        print(f"mAP50:     {row['mAP50']:.4f}")
        print(f"mAP50-95:  {row['mAP50_95']:.4f}")
        print(f"Training time: {row['training_minutes']:.2f} min")

        # Save partial CSV after every experiment
        csv_path = OUTPUT_DIR / "epoch_comparison.csv"

        with csv_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=results_table[0].keys()
            )
            writer.writeheader()
            writer.writerows(results_table)

    # Create comparison charts
    labels = [str(row["epochs_requested"]) for row in results_table]

    chart_metrics = [
        ("mAP50", "mAP50 by Epoch"),
        ("mAP50_95", "mAP50-95 by Epoch"),
        ("precision", "Precision by Epoch"),
        ("recall", "Recall by Epoch")
    ]

    for metric, title in chart_metrics:
        plt.figure(figsize=(8, 5))
        values = [row[metric] for row in results_table]

        plt.plot(labels, values, marker="o")
        plt.xlabel("Training Epochs")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        chart_path = OUTPUT_DIR / f"{metric}_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("           COMPARISON SUMMARY")
    print("=" * 60)
    print(
        f"{'Epoch':>8} {'Precision':>12} {'Recall':>10} "
        f"{'mAP50':>10} {'mAP50-95':>12}"
    )

    for row in results_table:
        print(
            f"{row['epochs_requested']:>8} "
            f"{row['precision']:>12.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['mAP50']:>10.4f} "
            f"{row['mAP50_95']:>12.4f}"
        )

    print(f"\nCSV and charts saved to: {OUTPUT_DIR}")
    print("Compare validation metrics to select a model.")
    print("Use the test set only for final evaluation.")


if __name__ == "__main__":
    main()