"""
mnist_lstm model training for MNIST recognition
Adapted from The TensorFlow authors
Modified by j-siderius
"""
import sys
import numpy as np
import tensorflow as tf

from tflm_converter import convert_tflite_to_tflm

def get_data():
    # Setup the dataset, using MNIST digit data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to 0-1
    x_train = x_train / 255.
    x_test = x_test / 255.

    # Set the data up as numpy floats
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test

def create_model(x_train) -> tf.keras.Model:
    # Create the TensorFlow LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=x_train[0].shape, name="input"),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax", name="output")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
        )

    model.summary()

    return model

def train_model(epochs, x_train, y_train, x_test, y_test):

    model = create_model(x_train)
    # Train the model using the data
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size=32
    )

    return model

def get_static_model(model, x_train):
    # Convert the model to have fixed input and output shapes
    fixed_input = tf.keras.layers.Input(
        shape=x_train[0].shape,
        batch_size=1,
        dtype=model.inputs[0].dtype,
        name="fixed_input"
        )
    fixed_output = model(fixed_input)

    static_model = tf.keras.models.Model(fixed_input, fixed_output)

    return static_model

def convert_tflite_model(model, x_train):
    static_model = get_static_model(model, x_train)

    # Save the model as TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
    tflite_model = converter.convert()

    return tflite_model


def main():
    print(f"Python version (should be 3.10.x or 3.11.x): {sys.version}")
    print(f"TensorFlow version (should be 2.15.1): {tf.__version__}")

    x_train, y_train, x_test, y_test = get_data()
    trained_model = train_model(5, x_train, y_train, x_test, y_test)

    tflite_model = convert_tflite_model(trained_model, x_train)
    open("trained_lstm.tflite", "wb").write(tflite_model)

    convert_tflite_to_tflm("trained_lstm.tflite")

if __name__ == "__main__":
    main()
