"""
hello_world model training for sinwave recognition
Adapted from The TensorFlow authors
Modified by j-siderius
"""
import math
import os

import numpy as np
import tensorflow as tf

from tools.tflm_converter import convert_tflite_to_tflm


def get_data():
  """
  The code will generate a set of random `x` values,calculate their sine
  values.
  """
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete sine wave oscillation
  x_values = np.random.uniform(low=0, high=2 * math.pi,
                               size=1000).astype(np.float32)

  # Shuffle the values to guarantee they're not in order
  np.random.shuffle(x_values)

  # Calculate the corresponding sine values
  y_values = np.sin(x_values).astype(np.float32)

  return (x_values, y_values)


def create_model() -> tf.keras.Model:
  model = tf.keras.Sequential()

  # First layer takes a scalar input and feeds it through 16 "neurons". The
  # neurons decide whether to activate based on the 'relu' activation function.
  model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(1, )))

  # The new second and third layer will help the network learn more complex
  # representations
  model.add(tf.keras.layers.Dense(16, activation='relu'))

  # Final layer is a single neuron, since we want to output a single value
  model.add(tf.keras.layers.Dense(1))

  # Compile the model using the standard 'adam' optimizer and the mean squared
  # error or 'mse' loss function for regression.
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])

  return model


def convert_tflite_model(model):
  """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
    Args:
        model (tf.keras.Model): the trained hello_world Model
    Returns:
        The converted model in serialized format.
  """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
  """save the converted tflite model
  Args:
      tflite_model (binary): the converted model in serialized format.
      save_dir (str): the save directory
      model_name (str): model name to be saved
  """
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
    print(f"TfLite model saved to {save_dir}")


def train_model(epochs, x_values, y_values):
  """Train keras hello_world model
    Args: epochs (int) : number of epochs to train the model
        x_train (numpy.array): list of the training data
        y_train (numpy.array): list of the corresponding array
    Returns:
        tf.keras.Model: A trained keras hello_world model
  """
  model = create_model()
  model.fit(x_values,
            y_values,
            epochs=epochs,
            validation_split=0.2,
            batch_size=64,
            verbose=2)

  return model


def main():
  x_values, y_values = get_data()
  trained_model = train_model(500, x_values, y_values)

  # Convert and save the model to .tflite
  tflite_model = convert_tflite_model(trained_model)
  save_tflite_model(tflite_model,
                    "./",
                    model_name="hello_world_float.tflite")

  convert_tflite_to_tflm("hello_world_float.tflite")


if __name__ == "__main__":
  main()

