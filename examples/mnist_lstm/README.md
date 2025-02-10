# MNIST LSTM example

This example is designed to demonstrate the application of an LSTM model using TFLite Micro. It includes a script to train and convert the model, as well as an example sketch which enables the depolyment to TFLite Micro on a microcontroller.

The model is trained to recognise handwritten digits (28x28 grayscale pixels) from the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist). The loop will randomly choose one of the ten included digits and predict the number that is drawn in it. While the MNIST dataset is not a typical scenario where an LSTM model is used in prediction, it gives a simple and easy to understand basis for further projects. Upload the example and open the Serial Monitor at baudrate 115200 to see the predictions.

Deploying a LSTM model onto a microcontroller with TFLite Micro involves a very specific setup. Newer versions of TensorFlow and Python do not correctly implement the TensorFlow Lite Converter to support LSTM layers. Some research[^1][^2] even suggests that LSTMs can be substituted for conventional CNN networks.

[^1]: S. Bai, J. Z. Kolter, and V. Koltun, ‘An empirical evaluation of generic convolutional and recurrent networks for sequence modeling’, Apr. 19, 2018, [arXiv: arXiv:1803.01271](https://arxiv.org/abs/1803.01271). doi: 10.48550/arXiv.1803.01271.
[^2]: X. Zhang, J. Zhao, and Y. LeCun, ‘Character-level Convolutional Networks for Text Classification’, Apr. 04, 2016, [arXiv: arXiv:1509.01626](https://arxiv.org/abs/1509.01626). doi: 10.48550/arXiv.1509.01626.

## Train your own model

> [!IMPORTANT]
> The training script requires **Python 3.10.x-3.11.x** and **TensorFlow 2.15.1** specifically. Using other versions will not generate a functioning TFLite Micro model.

Using the `train_mnist_lstm.py` file, a LSTM model that recognises handwritten digits from the MNIST dataset can be trained. The program does the following:

- Fetch the MNIST dataset from the TensorFlow repository.
- Build the TensorFlow Keras model with one LSTM layer, a Flattening layer and a Dense layer.
- Train the TensorFlow Keras model in five epochs.
- Convert the TensorFlow model into a TensorFlow Lite model and save it.
- Convert the TFLite model into a TFLite Micro model and save it.

After training, the model can be imported into the example Arduino sketch by either putting it in the source folder of the Arduino sketch or by including it using the 'Add Tab' option. The example can then be uploaded to the microcontroller.

## Logic of LSTMs on microcontrollers
The underlying logic that needs to be implemented in order to convert a TensorFlow LSTM layer into a TensorFlow Lite-compatible layer revolves around the need for fixed input and output shapes. In line with the [TensorFlow standard for RNN conversion](https://ai.google.dev/edge/litert/models/rnn#tensorflow_rnns_apis_supported), the model is locked into a state where the dimension 0 of the input tensor is the batch size (usually 1) and the the subsequent dimensions are the shape of a dataframe. By locking these shapes, TensorFlow Lite is able to assign fixed size input and output tensors to the model, which is nescessary in order to deploy the model to an edge device. After locking the shapes, the model can be converted using the normal TensorFlow Lite Converter as described in the [TensorFlow Lite Converter documentation](https://www.tensorflow.org/lite/models/convert/convert_models).

&copy; Jannick Siderius
