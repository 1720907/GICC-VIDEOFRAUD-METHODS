from keras.applications import VGG16, ResNet50, NASNetLarge, InceptionV3, DenseNet121

# Dictionary mapping model names to their corresponding Keras functions
MODEL_FUNCTIONS = {
    "model2_vgg16": VGG16,
    "model2_resnet": ResNet50,
    "model2_nasnet": NASNetLarge,
    "model2_inceptionv3": InceptionV3,
    "model2_densenet": DenseNet121
}

# Set of valid model names (for checking existence without functions)
VALID_MODELS = set(MODEL_FUNCTIONS.keys())
