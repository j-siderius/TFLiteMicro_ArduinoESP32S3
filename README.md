# TFLiteMicro_ArduinoESP32S3

Arduino library that ports TensorFlow Lite Micro to the ESP32-S3. Provides a very basic API for abstracting some of the TFLite c-specific references and nuances.

- Based on the [Espressif TFLite Micro repository](https://github.com/espressif/esp-tflite-micro).
- Utilises the optimised [ESP-NN](https://github.com/espressif/esp-nn) functions for the ESP32-S3 to speed up predictions.
- Three examples from Espressif and the [TensorFlow Lite Micro repository](https://github.com/tensorflow/tflite-micro) (Hello_World, Micro_Speech and MNIST_LSTM).
- Built for the [TinyML Development board](https://github.com/j-siderius/TinyML-board-documentation), but may work on other ESP32-S3 boards.
- Pre-compiled for the ESP32-S3 to speed up compilation.

## How to use

Also look at the [Minimal example](#minimal-example) for the implementation of TFLiteMicro_ArduinoESP32S3 into a sketch.

1. Gather input data using a (Python) simulation or Arduino sensor sketch.
2. Build and train a TensorFlow model in Python.
3. Save and convert the TensorFlow model into a TFLite (`.tflite`) model.
4. Convert the TFLite model using the `convert_tflite_to_tflm()` function from `tflm_converter.py`. Learn more in the [tools README](tools/README.md).
5. Import the generated model from `tflm_converter.py` into Arduino. Add it via the 'New Tab' option or copy it to the source folder of the Arduino sketch.
6. Include both the library and the model file at the top of the Arduino sketch, for example through `#include "TFLiteMicro_ArduinoESP32S3.h"` and `#include "example_model.h"`.
7. Initialise the model in the Arduino `setup()` by copying over the setup line from the model, for example `TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 5000>(TFLM_example_model, TFLMgetResolver);`.
8. Input data to the model by setting the input tensor(s), for example `TFLMinput->data.f[0] = x;`.
9. Let the model make a prediction by calling ` TFLMpredict();`.
10. Look at the output of the prediction by reading the output tensor(s), for example `float y = TFLMoutput->data.f[0];`.

## Library API
_tflite::MicroInterpreter_ **TFLMsetupModel**<int TFOperatorCount, size_t TFArenaSize>(const unsigned char *TFModel, tflite::MicroMutableOpResolver<TFOperatorCount> (*TFOperatorResolver)(), bool TFdebug = false)

Setup the TFLiteMicro model and initialise all dependencies. Schould generally be copied over from the generated model file.

| Parameter | Description |
| --- | --- |
| `TFOperatorCount` | The number of TFLiteMicro operators in the model (defined in the resolver header file) |
| `TFArenaSize` | The size (in bytes) of the model arena / working memory |
| `TFModel` | The TFLiteMicro model (defined in the resolver header file) |
| `TFOperatorResolver` | The resolver function for all TFLiteMicro operators (defined in the resolver header file) |
| `TFdebug` | Print some additional information during setup, default is `false` |


**Returns** &emsp;The TFLiteMicro interpreter pointer

<hr>

_bool_ **TFLMpredict**()

Predict an output from the given input(s), set them using for example `TFLMinput->data.f[a] = b`. Outputs can be read from the output tensor(s) using fore example `float a = TFLMoutput->data.f[a];`.

**Returns** &emsp;True if the prediction was successful, False if an error occurred

## Minimal example
```Arduino
// Include the library and example model
#include "TFLiteMicro_ArduinoESP32S3.h"
#include "example_model.h"

// Variable as input for the model
int a = 0;

void setup() {
    // Initialize the TensorFlow Lite Micro interpreter with the 
    // model and resolver (according to the command in the model header file)
    TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 5000>(TFLM_example_model, TFLMgetResolver);

    // Check if the model was set up correctly
    if (!TFLMinterpreter) {
        MicroPrintf("The model was not setup correctly.");
        for(;;){}
    }
}

void loop() {
    // Set the input data for the model
    TFLMinput->data.f[0] = a;

    // Perform inference using the model
    TFLMpredict();

    // Retrieve the output from the model
    float b = TFLMoutput->data.f[0];

    // Print the predicted value from the model
    MicroPrintf("The model predicted %d for input %d", b, a);
}
```

## Examples

[**Hello world**](examples/hello_world/)
<br>As the simplest TFLite Micro example, this program tries to recreate (predict) the values from a sine-wave function. The sketch can plot this wave in the Arduino Serial Plotter.

[**Micro speech**](examples/micro_speech/)
<br>This example recognises two keywords: `yes` and `no` as well as background noise. The model gives the probability for the most likely sound.

[**MNIST LSTM**](examples/mnsit_lstm/)
<br>This example uses the LSTM operation to classify handwritten digits from the MNIST dataset. Included in the example are 10 random numbers