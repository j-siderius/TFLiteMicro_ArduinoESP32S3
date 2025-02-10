# Hello world example

This example is designed to demonstrate the absolute basics of using TFLite Micro. It includes the full end-to-end workflow of training a model, converting it for use with TFLite Micro for running inference on a microcontroller.

The model is trained to replicate a `sine` function and can draw the function in the Arduino Serial Plotter. Upload the example and open the Serial Plotter at baudrate 115200 to see the sine function.

## Train your own model

Using the `train_sine.py` file, a model that predicts the values of sine can be trained. The program does the following:

- Generate 1000 synthetic datapoints from the sine function (x-y pairs).
- Build the TensorFlow Keras model with two hidden Dense layers of 16 neurons.
- Train the TensorFlow Keras model in 500 epochs.
- Convert the TensorFlow model into a TensorFlow Lite model and save it.
- Convert the TFLite model into a TFLite Micro model and save it.

After training, the model can again be imported into the Arduino sketch by either putting it in the source folder of the Arduino sketch or by including it using the 'Add Tab' option.

&copy; Jannick Siderius
