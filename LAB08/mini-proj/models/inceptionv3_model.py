from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19

IMG_SIZE = (224, 224, 3)

def build_model():
    base = VGG19(weights="imagenet", include_top=False, input_shape=IMG_SIZE)
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="VGG19")
