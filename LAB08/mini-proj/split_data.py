from pathlib import Path
import shutil
import random

# Dataset ของคุณ
DATA_DIR = Path("PetImages")

# ที่เก็บ Dataset ที่แบ่งแล้ว
OUTPUT_DIR = Path("split_data")

# สัดส่วน
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif"
}


def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():

    random.seed(SEED)

    # ใช้ชื่อโฟลเดอร์ตาม Dataset จริงของคุณ
    classes = {
        "cat": "cat",
        "dog": "dog"
    }

    for class_name, folder_name in classes.items():

        source = DATA_DIR / folder_name

        if not source.exists():
            raise FileNotFoundError(
                f"ไม่พบโฟลเดอร์: {source}"
            )

        images = get_images(source)

        print(f"\n{class_name.upper()}")
        print(f"พบรูปทั้งหมด: {len(images)}")

        # สุ่มรูป
        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]

        val_images = images[
            train_count:
            train_count + val_count
        ]

        test_images = images[
            train_count + val_count:
        ]

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split_name, split_images in splits.items():

            destination = (
                OUTPUT_DIR
                / split_name
                / class_name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for image in split_images:

                target = destination / image.name

                shutil.copy2(
                    image,
                    target
                )

            print(
                f"{split_name:5} = "
                f"{len(split_images)} รูป"
            )

    print("\n" + "=" * 50)
    print("แบ่ง Dataset สำเร็จ")
    print("=" * 50)

    print("\nโครงสร้างที่ได้:")

    print("""
split_data/
├── train/
│   ├── cat/
│   └── dog/
│
├── val/
│   ├── cat/
│   └── dog/
│
└── test/
    ├── cat/
    └── dog/
""")


if __name__ == "__main__":
    main()