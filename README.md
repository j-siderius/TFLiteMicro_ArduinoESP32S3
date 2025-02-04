# TFLiteMicro_ArduinoESP32S3

Arduino library that ports TensorFlow Lite Micro to the ESP32-S3. Provides a very basic API for abstracting some of the TFLite c-specific references and nuances.

- Based on the [Espressif TFLite Micro repository](https://github.com/espressif/esp-tflite-micro).
- Utilises the optimised [ESP-NN](https://github.com/espressif/esp-nn) functions for the ESP32-S3 to speed up predictions considerably.
- Three examples from Espressif and the [TensorFlow Lite Micro repository](https://github.com/tensorflow/tflite-micro) (Hello_World, Micro_Speech and MNIST_LSTM).
- Written specifically for the [TinyML Development board](https://github.com/j-siderius/TinyML-board-documentation), but may work on other ESP32-S3 boards [^1].
- Pre-compiled for the ESP32-S3 to speed up compilation significantly.

[^1]: All operations should be supported on every ESP32-S3 board, however some pin-specific configuration (for example in the Micro_Speech example sketch) may differ between ESP32-S3 boards. Some boards may also feature less Flash and (PS)RAM storage than the TinyML Development Board, meaning there is less space to store models.

## Contents

- [How to use](#how-to-use)
- [Library API](#library-api)
    - [TFLMsetupModel](#tflmsetupmodelint-tfoperatorcount-size_t-tfarenasizeconst-unsigned-char-tfmodel-tflitemicromutableopresolver-tfoperatorresolver-bool-tfdebug--false)
    - [TFLMpredict](#tflmpredict)
    - [TFLMinterpreter](#tflminterpreter)
    - [TFLMinput](#tflminput)
    - [TFLMoutput](#tflmoutput)
- [Minimal code example](#minimal-code-example)
- [Examples](#examples)
    - [Hello world](#hello-world)
    - [Micro speech](#micro-speech)
    - [MNIST LSTM](#mnist-lstm)
- [Advanced options](#advanced-options)
    - [Reducing required model memory](#reducing-required-model-memory)
    - [Input and Output testing](#input-and-output-testing)

## How to use

The [Minimal example](#minimal-example) gives a quick overview for the implementation of TFLiteMicro_ArduinoESP32S3 into a sketch. Below are all the steps nescessary to go from data to a TensorFlow model to a TFLite Micro model running on the ESP32-S3.

1. Gather input data using a (Python) simulation or Arduino sensor sketch.
2. Build and train a TensorFlow model in Python.
3. Save and convert the TensorFlow model into a TFLite (`.tflite`) model.
4. Convert the TFLite model using the `convert_tflite_to_tflm()` function from `tflm_converter.py`. Learn more in the [tools README](tools).
5. Import the generated model from `tflm_converter.py` into Arduino. Add it via the 'New Tab' option or copy it to the source folder of the Arduino sketch.
6. Include both the library and the model file at the top of the Arduino sketch, for example through `#include "TFLiteMicro_ArduinoESP32S3.h"` and `#include "example_model.h"`.
7. Initialise the model in the Arduino `setup()` by copying over the setup line from the model, for example `TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 5000>(TFLM_example_model, TFLMgetResolver);`.
8. Input data to the model by setting the input tensor(s), for example `TFLMinput->data.f[0] = x;`.
9. Let the model make a prediction by calling ` TFLMpredict();`.
10. Look at the output of the prediction by reading the output tensor(s), for example `float y = TFLMoutput->data.f[0];`.

## Library API

#### TFLMsetupModel<int TFOperatorCount, size_t TFArenaSize>(const unsigned char *TFModel, tflite::MicroMutableOpResolver<TFOperatorCount> (*TFOperatorResolver)(), bool TFdebug = false)

**Type** &emsp;Function - _tflite::MicroInterpreter generator_

Setup the TFLiteMicro model and initialise all dependencies. Schould generally be copied over from the generated model file.

| Parameter | Description |
| --- | --- |
| `TFOperatorCount` | The number of TFLiteMicro operators in the model (defined in the resolver header file) |
| `TFArenaSize` | The size (in bytes) of the model arena / working memory |
| `TFModel` | The TFLiteMicro model (defined in the resolver header file) |
| `TFOperatorResolver` | The resolver function for all TFLiteMicro operators (defined in the resolver header file) |
| `TFdebug` | Debug some additional information during setup, default is `false`. The debugger prints the actual size (in bytes) of the model arena, as well as the time it took to make the prediction (in microseconds). |


**Returns** &emsp;The TFLiteMicro interpreter reference

#### TFLMpredict()

**Type** &emsp;Function - _bool_

Predict an output from the given input(s), set them using for example `TFLMinput->data.f[a] = b`. Outputs can be read from the output tensor(s) using for example `float a = TFLMoutput->data.f[a];`. For more information about the inputs and outputs of the model, refer to the [TFLMinput](#tflminput) and [TFLMoutput](#tflmoutput) sections.

| Parameter | Description |
| --- | --- |
| N/A |  |

**Returns** &emsp;_True_ if the prediction was successful, _False_ if an error occurred

<br>

#### TFLMinterpreter

**Type** &emsp;Object - _tflite::MicroInterpreter_

The TFLite Micro interpreter reference, which handles all inputs, inference (predictions) and outputs. The `TFLMinterpreter` is initialised and stored inside of the library. The interpreter can be checked by querrying `if (TFLMinterpreter)` which should equate to _True_ if the model was setup correctly.

<br>

#### TFLMinput

**Type** &emsp;Variable - _TfLiteTensor_

The TFLite Micro input array pointer, which handles all inputs for the model. Inputs can be of different variable types (for example _uint8_t_, _float_ or _double_), depending on your TensorFlow model. To set an input to the model, point the input to the correct datatype and array position and assign it the value. For example, to set the second input `TFLMinput->data.f[1] = 16.4;`. In the table below are some of the most common datatypes. If the input consists of a matrix (or array), all dimensions will get squashed into one:
```
[[a b c]
 [d e f]] = [a b c d e f]
```

#### TFLMoutput

**Type** &emsp;Variable - _TfLiteTensor_

The TFLite Micro output array pointer, which stores all outputs from the model. Output can be of different variable types, just like inputs, depending on the TensorFlow model. To read from an output of the model, assign the value of the output to a variable in the correct datatype. For example, to read the third output `float a = TFLMoutput->data.f[2];`. The same datatypes apply as for the input, see the table below. Just like the input, the output will get squashed into one dimension, regardless of the initial shape.


| Input/Output Datatype | Code |
| --- | --- |
| | <small>`i` is the array index of the input or output</small>  |
| int8_t (signed 8-bit integer) | `data.int8[i]` |
| uint8_t (unsigned 8-bit integer) | `data.uint8[i]` |
| float | `data.f[i]`|
| bool | `data.b[i]` |
| char | `data.raw[i]` |

*Other datatypes can be found in the [common.h _TfLitePtrUnion_](https://github.com/j-siderius/TFLiteMicro_ArduinoESP32S3/blob/main/src/tensorflow/lite/core/c/common.h#L361) definition.*

## Minimal code example
```cpp
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

#### [Hello world](/examples/hello_world/)

As the simplest TFLite Micro example, this program tries to recreate (predict) the values from a sine-wave function. The sketch can plot this wave in the Arduino Serial Plotter.

#### [Micro speech](/examples/micro_speech/)

This example recognises two keywords: `yes` and `no` as well as background noise. The model gives the probability for the most likely sound.

#### [MNIST LSTM](/examples/mnist_lstm/)

This example uses the LSTM operation to classify handwritten digits from the MNIST dataset. Included in the example are 10 random handwritten numbers, stored as byte arrays

## Advanced options

#### Reducing required model memory

The setup command for the TFLiteMicro_ArduinoESP32S3 library, generated using the tflm_converter tool can be modified in order to reduce the memory requirement. The default `TFArenaSize` is generated using a simple algorithm[^2], however it can be reduced to reduce the total memory footprint. To see the actual required memory:

[^2]: The algorithm simply looks at the final `model_length`, then rounds up to the closest 5000 bytes. Look at the implementation in the [tflm_convert.py program](https://github.com/j-siderius/TFLiteMicro_ArduinoESP32S3/blob/main/tools/tflm_converter.py#L171).

1. Upload the model using the default calculated `TFArenaSize`, but enabling the `TFdebug=true` option in the model setup function, for example `TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 5000>(TFLM_example_model, TFLMgetResolver, true);`.
2. Look at the serial output of the model, the debugger will respond with something like `DEBUG:Tensor arena used 2440 bytes`.
3. The `TFArenaSize` can now be changed to a value closer to the actual (allocated) tensor arena size. _It is advisable to always give the model a bit more memory than the absolute minimum._
4. Change the model setup function, for example `TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 2500>(TFLM_example_model, TFLMgetResolver, true);`, and upload the model.

#### Input and Output testing

In some cases it might be useful to know the dimension (shape) and type of the input or outputs (TfliteTensors).

To check the dimension of the tensor, query it by calling `TFLMinput->dims->size` which returns the amount of elements in the input or output.

To check the datatype of the tensor, query it by calling `TFLMoutput->type` which returns the type of the input or output. The output can be decyphered in the [tflite_types.h type definition](https://github.com/j-siderius/TFLiteMicro_ArduinoESP32S3/blob/main/src/tensorflow/compiler/mlir/lite/core/c/tflite_types.h#L46).

To check the parameters of the tensor, query it by calling `TFLMinput->params.scale` or `TFLMoutput->params.zero_point` which return the quantisation parameters of the input or output.

&copy; Jannick Siderius 
