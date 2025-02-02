#pragma once

#include "Arduino.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
    const tflite::Model *_model = nullptr; // Internal pointer to the TFLite model
    tflite::MicroInterpreter *interpreter = nullptr; // Pointer to the interpreter
    TfLiteTensor *input = nullptr; // Pointer to the input tensor
    TfLiteTensor *output = nullptr; // Pointer to the output tensor
} // namespace

/*!
    @brief  Setup the TFLiteMicro model and initialise all dependencies
    @tparam TFOperatorCount   The number of TFLiteMicro operators in the model (defined in the resolver header file)
    @tparam TFArenaSize The size (in bytes) of the model arena / working memory
    @param  TFModel     The TFLiteMicro model (defined in the resolver header file)
    @param  TFOperatorResolver  The resolver function for all TFLiteMicro operators (defined in the resolver header file)
    @return The TFLiteMicro interpreter pointer 
*/
template <int TFOperatorCount, size_t TFArenaSize>
tflite::MicroInterpreter* setupModel(const unsigned char *TFModel,
                tflite::MicroMutableOpResolver<TFOperatorCount> (*TFOperatorResolver)())
{

    static_assert(TFOperatorCount > 0, "The number of operators in the model is 0 (which seems wrong).");
    static_assert(TFArenaSize > 100, "The arenasize is too small (<100 bytes).");

    static uint8_t _tensor_arena[TFArenaSize]; // Internal memory arena for tensor allocation

    _model = tflite::GetModel(TFModel); // Retrieve the model from the provided data
    if (_model->version() != TFLITE_SCHEMA_VERSION)
    {
        MicroPrintf("Model provided is schema version %d not equal to supported version %d.",
                    _model->version(), TFLITE_SCHEMA_VERSION);
        return nullptr; // Return null if the model version is unsupported
    }

    // Initialize resolver with the correct number of ops
    static tflite::MicroMutableOpResolver<TFOperatorCount> resolver = TFOperatorResolver();

    // Build an internal interpreter to run the model with.
    static tflite::MicroInterpreter _static_interpreter(
        _model, resolver, _tensor_arena, TFArenaSize);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = _static_interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        MicroPrintf("AllocateTensors() failed, the arenasize is (probably) too small.");
        return nullptr; // Return null if tensor allocation fails
    }

    return &_static_interpreter; // Return the interpreter pointer
}

/*!
    @brief  Predict an output from the given input(s), set them using `interpreter->input[x] = y`
    @return True if the prediction was successful, False if an error occurred
*/
bool predictModel()
{
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk)
    {
        MicroPrintf("Invoke failed."); // Log failure of the invoke operation
        return false; // Return false if invocation fails
    }

    return true; // Return true if prediction was successful
}
