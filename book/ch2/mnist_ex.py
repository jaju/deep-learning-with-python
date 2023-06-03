from tensorflow import keras
from keras import layers
from keras.datasets import mnist


def init_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_mnist():
    return mnist.load_data()


def reshape_mnist(train_data, test_data):
    (train_images, train_labels) = train_data
    (test_images, test_labels) = test_data
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    return (train_images, train_labels), (test_images, test_labels)
