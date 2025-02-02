#include "TFLiteMicro_ArduinoESP32S3.h"
#include "hello_world_float_model.h"

// Global variables
int InferenceCount = 0; // Counter for the number of inferences performed
const float Xrange = 2.f * 3.14159265359f; // Range of x values (0 to 2π)
const float InferencesPerCycle = 20; // Number of inferences to perform in one cycle

void setup()
{
  // Initialize the TensorFlow Lite Micro interpreter with the model and resolver (according to the command in the model header file)
  TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 10000>(TFLM_hello_world_float_model, TFLMgetResolver);
  
  // Check if the model was set up correctly
  if (!TFLMinterpreter) {
    MicroPrintf("The model was not setup correctly.");
    for(;;){}
  }

  // Initialize inference count
  InferenceCount = 0;
}

void loop()
{
  // Calculate the current position within the range of x values
  float position = InferenceCount / InferencesPerCycle; // Normalized position
  float x = position * Xrange; // Calculate x value based on position

  // Set the input data for the model
  TFLMinput->data.f[0] = x;

  // Perform inference using the model
  TFLMpredict();

  // Retrieve the output from the model
  float y = TFLMoutput->data.f[0];

  // Log the current x and output y values
  MicroPrintf("x_value:%f,y_value:%f", x, y);

  // Increment the inference counter and reset if it reaches the cycle limit
  InferenceCount += 1;
  if (InferenceCount >= InferencesPerCycle)
    InferenceCount = 0; // Reset counter for the next cycle

  // Delay to control the frequency of inferences (100ms)
  delay(100);
}
