from pathlib import Path
import random
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "runs" / "aquarium" / "weights" / "best.pt"
IMAGE_DIR = ROOT / "dataset" / "test" / "images"
OUTPUT_DIR = ROOT / "runs" / "image_test"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
NUM_IMAGES = 5
DEVICE = 0


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    model = YOLO(str(MODEL_PATH))

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not images:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    random.seed(42)
    selected = random.sample(images, min(NUM_IMAGES, len(images)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Testing {len(selected)} images...")

    try:
        for index, image_path in enumerate(selected, start=1):
            print(f"\n[{index}/{len(selected)}] {image_path.name}")

            result = model.predict(
                source=str(image_path),
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )[0]

            if result.boxes is None or len(result.boxes) == 0:
                print("No objects detected.")
            else:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    print(
                        f"{model.names[class_id]}: {confidence:.2%}"
                    )

            annotated = result.plot()
            output_path = OUTPUT_DIR / f"{index}_{image_path.name}"

            if not cv2.imwrite(str(output_path), annotated):
                print(f"Could not save image: {output_path}")
            else:
                print(f"Saved: {output_path}")

            cv2.imshow("Aquarium YOLOv8 Detection", annotated)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()