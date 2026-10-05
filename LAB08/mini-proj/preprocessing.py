def get_preprocessing(model_name):
    model_name = model_name.lower()

    if model_name in ["vgg16", "vgg19"]:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input

    if model_name == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input

    if model_name == "resnet101":
        from tensorflow.keras.applications.resnet import preprocess_input
        return preprocess_input

    if model_name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input

    if model_name == "efficientnetb0":
        return lambda x: x

    raise ValueError(f"ไม่รู้จัก model: {model_name}")
