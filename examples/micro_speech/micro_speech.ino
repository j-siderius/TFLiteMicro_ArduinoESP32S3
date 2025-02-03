#include "TFLiteMicro_ArduinoESP32S3.h"
#include "micro_speech_quantized_model.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"

// Globals
namespace
{
  FeatureProvider *feature_provider = nullptr; // Pointer to the feature provider
  int32_t previous_time = 0; // Timestamp of the last processed audio

  // Buffer for storing feature data
  int8_t feature_buffer[kFeatureElementCount]; // Intermediate array for features
  int8_t *model_input_buffer = nullptr; // Pointer to the model's input buffer
} // namespace

void setup()
{
  // Initialize the TensorFlow Lite Micro interpreter with the quantized model
  TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 19000>(TFLM_micro_speech_quantized_model, TFLMgetResolver);

  // Check if the model was set up correctly
  if (!TFLMinterpreter) {
    MicroPrintf("The model was not setup correctly.");
    for(;;){} // Infinite loop if setup fails
  }

  // Validate input tensor parameters for the model
  if ((TFLMinput->dims->size != 2) || (TFLMinput->dims->data[0] != 1) ||
      (TFLMinput->dims->data[1] != (kFeatureCount * kFeatureSize)) ||
      (TFLMinput->type != kTfLiteInt8))
  {
    MicroPrintf("Bad input tensor parameters in model");
    return; // Exit setup if parameters are invalid
  }
  
  // Get a pointer to the model's input tensor data
  model_input_buffer = tflite::GetTensorData<int8_t>(TFLMinput);

  // Prepare to access audio spectrograms from a microphone or other source
  static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
  feature_provider = &static_feature_provider; // Assign the feature provider

  previous_time = 0; // Initialize previous time
}

void loop()
{
  // Fetch the current audio timestamp
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0; // Counter for new audio slices

  // Populate feature data from the audio provider
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk)
  {
    MicroPrintf("Feature generation failed");
    return; // Exit loop if feature generation fails
  }
  
  previous_time = current_time; // Update previous time

  // If no new audio samples have been received, skip model inference
  if (how_many_new_slices == 0)
  {
    return;
  }

  // Copy feature data to the model's input tensor
  for (int i = 0; i < kFeatureElementCount; i++)
  {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Perform inference using the model
  TFLMpredict();

  // Obtain output tensor parameters for dequantization
  float output_scale = TFLMoutput->params.scale;
  int output_zero_point = TFLMoutput->params.zero_point;
  int max_idx = 0; // Index of the category with the highest score
  float max_result = 0.0; // Highest score found

  // Dequantize output values and find the maximum score
  for (int i = 0; i < kCategoryCount; i++)
  {
    float current_result =
        (tflite::GetTensorData<int8_t>(TFLMoutput)[i] - output_zero_point) *
        output_scale;
    if (current_result > max_result)
    {
      max_result = current_result; // Update max result
      max_idx = i;                 // Update category index
    }
  }

  // Log the detected category if the score exceeds the threshold
  if (max_result > 0.8f)
  {
    MicroPrintf("Detected %7s, score: %.2f", kCategoryLabels[max_idx],
                static_cast<double>(max_result));
  } else {
    MicroPrintf("Only background noise detected."); // Log if only noise is detected
  }
}
