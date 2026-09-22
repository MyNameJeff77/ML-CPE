import os

import cv2
import numpy as np

from preprocessing import preprocess_image


VALID_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


def load_data(
    data_path,
    img_size=100,
    max_per_class=None
):

    # =========================================================
    # CHECK DATASET
    # =========================================================

    if not os.path.isdir(data_path):

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{os.path.abspath(data_path)}\n\n"
            f"Expected structure:\n"
            f"PetImages/\n"
            f"├── Cat/\n"
            f"└── Dog/\n"
        )

    images = []
    labels = []

    # =========================================================
    # DETECT CLASSES
    # =========================================================

    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(
            os.path.join(
                data_path,
                folder
            )
        )
    ])

    print(
        "\nDetected classes:",
        classes
    )

    if len(classes) != 2:

        raise ValueError(
            f"Expected exactly 2 classes "
            f"(Cat and Dog), found: {classes}"
        )

    # =========================================================
    # LOAD EACH CLASS
    # =========================================================

    for label, class_name in enumerate(classes):

        class_path = os.path.join(
            data_path,
            class_name
        )

        filenames = sorted([
            f
            for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        ])

        loaded = 0
        skipped = 0

        print(
            f"\nLoading {class_name}..."
        )

        for filename in filenames:

            # Limit images per class
            if (
                max_per_class is not None
                and loaded >= max_per_class
            ):
                break

            image_path = os.path.join(
                class_path,
                filename
            )

            # -------------------------------------------------
            # READ IMAGE
            # -------------------------------------------------

            image = cv2.imread(
                image_path
            )

            if image is None:

                skipped += 1

                continue

            # -------------------------------------------------
            # PREPROCESS
            # -------------------------------------------------

            image = preprocess_image(
                image,
                img_size
            )

            if image is None:

                skipped += 1

                continue

            images.append(
                image
            )

            labels.append(
                label
            )

            loaded += 1

        print(
            f"Loaded {class_name}: "
            f"{loaded} images"
        )

        print(
            f"Skipped: {skipped}"
        )

    # =========================================================
    # CHECK RESULT
    # =========================================================

    if len(images) == 0:

        raise ValueError(
            "No valid images were loaded."
        )

    # =========================================================
    # CONVERT TO NUMPY
    # =========================================================

    images = np.stack(
        images
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    print("\nDataset loaded successfully.")

    print(
        f"Image shape : {images.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    return (
        images,
        labels,
        classes
    )