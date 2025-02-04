# Micro Speech example

This example shows how to run inference using TensorFlow Lite Micro (TFLM) on two models for wake-word recognition. The first model is an audio preprocessor that generates spectrogram data from raw audio samples. The second is the Micro Speech model, a less than 20 kB model that can recognize 2 keywords, "yes" and "no", from speech data. The Micro Speech model takes the spectrogram data as input and produces category probabilities.

The model is trained to recognise `yes`, `no` and `background noise` and print the most probably keyword to the Arduino Serial Console. Upload the example, open the Serial Console at 115200 baudrate and speak a keyword to see the model work.

## More information

More information on how the model was built, as well as the steps necessary to train your own model can be found in the [TfLite-Micro Micro Speech example README](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech) and the [Train your own Micro Speech recogniser README](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech/train).

## Deploying your own model

After training the model, it again be converted into a TFLite and subsequently a TFLite Micro model using the [TFLM tools](/tools/). The model can then be imported into the Arduino sketch by either putting it in the source folder of the Arduino sketch or by including it using the 'Add Tab' option.

&copy; Jannick Siderius
