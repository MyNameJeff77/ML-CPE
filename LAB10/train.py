from pathlib import Path
from ultralytics import YOLO
import torch
import time

ROOT = Path(__file__).resolve().parent

DATA_YAML = ROOT / "dataset" / "data.yaml"
MODEL_NAME = "yolov8n.pt"

EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0

PROJECT = ROOT / "runs"
RUN_NAME = "aquarium"


def main():
    print("=" * 55)
    print("       AQUARIUM OBJECT DETECTION - YOLOv8")
    print("=" * 55)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check your PyTorch installation."
        )

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Dataset : {DATA_YAML}")
    print(f"Epochs  : {EPOCHS}")
    print(f"Image   : {IMAGE_SIZE}")
    print(f"Batch   : {BATCH_SIZE}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}"
        )

    model = YOLO(MODEL_NAME)
    start_time = time.time()

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        amp=True,
        cache=False,
        patience=10,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        save=True,
        save_period=5,
        plots=True,
        seed=42,
        verbose=True
    )

    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)
    print(f"Time       : {(time.time() - start_time) / 60:.2f} min")
    print(f"Best model : {best_path}")
    print(f"Last model : {last_path}")
    print(f"Results    : {save_dir / 'results.csv'}")
    print(f"Plots      : {save_dir}")
    print("=" * 55)

    if not best_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_path}")

    print("\nEvaluating best model on validation set...")

    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(PROJECT),
        name="validation",
        exist_ok=True
    )

    print("\n===== VALIDATION RESULTS =====")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()