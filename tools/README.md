# TFLiteMicro tools

These tools provide an easy way to convert a TFLite model in Python into a TFLite Micro model that is ready to import into Arduino.

The converter checks if all operations in the TFLite model are compatible with TFLite Micro using the `micro_mutable_op_resolver.h` reference file. Then all operations are extracted from the TFLite model, and the parameters (weights, biases and activations) are converted into Arduino-compatible hex-format. Lastly, the TFLite Micro model is built using a template, which also generates the correct `TFLMsetupModel` function.

To add the generated model to an Arduino sketch, either place it in to the Arduino project folder of the main sketch, or add it via the 'New Tab' option.
To use the generated model, include the setup line from the model, for example `TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 5000>(TFLM_example_model, TFLMgetResolver);`.

## Contents

- [Requirements](#requirements)
- [Library API](#library-api)
- [Minimal code example](#minimal-code-example)

## Requirements

The tools require the following dependencies:

| Library | Version | Link |
| --- | --- | --- |
| Mako | >=1.3.8 | https://pypi.org/project/Mako/ |
| TensorFlow | >=2.18.0 | https://pypi.org/project/tensorflow/ |

## Library API

#### convert_tflite_to_tflm(tflite_path: str = "model.tflite") -> str

**Type** &emsp;Function - _model generator_

Convert the specified TFLite model into a TFLite Micro header file to be used in Arduino together with the TFLiteMicro_ArduinoESP32S3 library.

| Parameter | Description |
| --- | --- |
| `tflite_path` | Path to the TFLite model file, the name of this model will determine the final header file name |

**Returns** &emsp;String-path to the converted Arduino header file

## Minimal code example
```python
# Import the converter tool
from tools.tflm_converter import convert_tflite_to_tflm

# Generate the TFLite model here
# ---

# Convert the TFLite model
convert_tflite_to_tflm("example_model.tflite")
```

&copy; Jannick Siderius 
