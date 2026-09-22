import json
import os

from tensorflow import keras
from tensorflow.keras import layers, regularizers


# =========================================================
# BUILD CNN MODEL
# =========================================================

def build_model(
    input_shape,
    num_classes
):

    model = keras.Sequential([

        # -----------------------------------------------------
        # INPUT
        # -----------------------------------------------------

        keras.Input(
            shape=input_shape
        ),

        # -----------------------------------------------------
        # NORMALIZATION
        # -----------------------------------------------------

        layers.Rescaling(
            1.0 / 255
        ),

        # -----------------------------------------------------
        # DATA AUGMENTATION
        # -----------------------------------------------------

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        # =====================================================
        # CNN BLOCK 1
        # =====================================================

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        # =====================================================
        # CNN BLOCK 2
        # =====================================================

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        # =====================================================
        # CNN BLOCK 3
        # =====================================================

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        # =====================================================
        # FEATURE EXTRACTION
        # =====================================================

        layers.GlobalAveragePooling2D(),

        # =====================================================
        # CLASSIFIER
        # =====================================================

        layers.Dropout(
            0.30
        ),

        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),

        layers.Dropout(
            0.30
        ),

        # =====================================================
        # OUTPUT
        # =====================================================

        layers.Dense(
            1 if num_classes == 2 else num_classes,

            activation=(
                "sigmoid"
                if num_classes == 2
                else "softmax"
            )
        )
    ])

    # =========================================================
    # COMPILE
    # =========================================================

    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss=(
            "binary_crossentropy"
            if num_classes == 2
            else "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]
    )

    return model


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    output_dir=None,
    epochs=50,
    batch_size=32
):

    # ---------------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------------

    model = build_model(
        X_train.shape[1:],
        num_classes
    )

    print("\n")
    model.summary()

    # =========================================================
    # CALLBACKS
    # =========================================================

    callbacks = []

    # ---------------------------------------------------------
    # Reduce Learning Rate
    # ---------------------------------------------------------

    callbacks.append(

        keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=1e-6,

            verbose=1
        )
    )

    # ---------------------------------------------------------
    # Early Stopping
    # ---------------------------------------------------------

    callbacks.append(

        keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=10,

            restore_best_weights=True,

            verbose=1
        )
    )

    # ---------------------------------------------------------
    # Save Best Model
    # ---------------------------------------------------------

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        best_model_path = os.path.join(
            output_dir,
            "best_model.keras"
        )

        callbacks.append(

            keras.callbacks.ModelCheckpoint(

                best_model_path,

                monitor="val_loss",

                save_best_only=True,

                verbose=1
            )
        )

    # =========================================================
    # TRAIN
    # =========================================================

    print("\n")
    print("=" * 60)
    print("Training...")
    print("=" * 60)

    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=epochs,

        batch_size=batch_size,

        callbacks=callbacks,

        shuffle=True,

        verbose=1
    )

    # =========================================================
    # SAVE MODEL
    # =========================================================

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        model_path = os.path.join(
            output_dir,
            "cnn_model.keras"
        )

        model.save(
            model_path
        )

        # -----------------------------------------------------
        # SAVE HISTORY
        # -----------------------------------------------------

        history_path = os.path.join(
            output_dir,
            "history.json"
        )

        history_dict = {

            key: [
                float(value)
                for value in values
            ]

            for key, values
            in history.history.items()
        }

        with open(
            history_path,
            "w"
        ) as f:

            json.dump(
                history_dict,
                f,
                indent=4
            )

        print(
            f"\nSaved model: {model_path}"
        )

        print(
            f"Saved history: {history_path}"
        )

    return (
        model,
        history
    )


# =========================================================
# PREDICT
# =========================================================

def predict_model(
    model,
    X_test
):

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    # ---------------------------------------------------------
    # Binary Classification
    # ---------------------------------------------------------

    if probabilities.shape[-1] == 1:

        return (
            probabilities.ravel() >= 0.5
        ).astype(int)

    # ---------------------------------------------------------
    # Multi-class
    # ---------------------------------------------------------

    return probabilities.argmax(
        axis=1
    )