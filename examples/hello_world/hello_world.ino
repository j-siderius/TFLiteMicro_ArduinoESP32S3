/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "TFLiteMicro_ArduinoESP32S3.h"
#include "hello_world_float_resolver.h"

int inference_count = 0;
const float kXrange = 2.f * 3.14159265359f;
const float kInferencesPerCycle = 20;

void setup()
{

  interpreter = setupModel<kNumberOperators, 2000>(hello_world_float, get_resolver);
  if (!interpreter) {
    MicroPrintf("The model was not setup correctly.");
    for(;;){}
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = inference_count / kInferencesPerCycle;
  float x = position * kXrange;

  input->data.f[0] = x;

  predictModel();

  float y = output->data.f[0];

  // Log the current X and output Y
  MicroPrintf("x_value:%f,y_value:%f", x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle)
    inference_count = 0;

  // trigger one inference every 100ms
  delay(100);
}
