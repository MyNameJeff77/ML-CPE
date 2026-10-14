from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "runs" / "epoch_comparison"
    / "aquarium_30epochs" / "weights" / "best.pt"
)
DATA_YAML = ROOT / "dataset" / "data.yaml"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_PATH}")
    print("Evaluating on TEST dataset...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / "runs"),
        name="aquarium_30epochs_test",
        exist_ok=True
    )

    print("\n===== FINAL TEST RESULTS: 30 EPOCHS =====")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()