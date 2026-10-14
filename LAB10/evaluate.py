from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "runs" / "aquarium" / "weights" / "best.pt"
DATA_YAML = ROOT / "dataset" / "data.yaml"

IMAGE_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0


def main():
    print("=" * 55)
    print("       AQUARIUM MODEL EVALUATION")
    print("=" * 55)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train.py first."
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}"
        )

    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Model : {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(ROOT / "runs"),
        name="evaluation",
        exist_ok=True
    )

    print("\n===== TEST RESULTS =====")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()