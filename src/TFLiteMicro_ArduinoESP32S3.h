#pragma once

#include "Arduino.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/core/c/common.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
    const tflite::Model *_model = nullptr;               // Internal pointer to the TFLite model
    tflite::MicroInterpreter *TFLMinterpreter = nullptr; // Pointer to the interpreter
    TfLiteTensor *TFLMinput = nullptr;                   // Pointer to the input tensor
    TfLiteTensor *TFLMoutput = nullptr;                  // Pointer to the output tensor
    unsigned long _previousMicros = 0;                   // For timing the inference speed
    bool _debug = false;                                 // If the functions should print debug information
} // namespace

/*!
    @brief  Setup the TFLiteMicro model and initialise all dependencies
    @tparam TFOperatorCount   The number of TFLiteMicro operators in the model (defined in the resolver header file)
    @tparam TFArenaSize The size (in bytes) of the model arena / working memory
    @param  TFModel     The TFLiteMicro model (defined in the resolver header file)
    @param  TFOperatorResolver  The resolver function for all TFLiteMicro operators (defined in the resolver header file)
    @param  TFdebug Print some additional information during setup, default is false
    @return The TFLiteMicro interpreter pointer
*/
template <int TFOperatorCount, size_t TFArenaSize>
tflite::MicroInterpreter *TFLMsetupModel(const unsigned char *TFModel,
                                         tflite::MicroMutableOpResolver<TFOperatorCount> (*TFOperatorResolver)(), bool TFdebug = false)
{

    static_assert(TFOperatorCount > 0, "The number of operators in the model is 0 (which seems wrong).");
    static_assert(TFArenaSize > 100, "The arenasize is too small (<100 bytes).");

    _debug = TFdebug;

    static uint8_t _tensor_arena[TFArenaSize]; // Internal memory arena for tensor allocation

    _model = tflite::GetModel(TFModel); // Retrieve the model from the provided data
    if (_model->version() != TFLITE_SCHEMA_VERSION)
    {
        MicroPrintf("TFModel provided is schema version %d not equal to supported version %d.",
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
        MicroPrintf("AllocateTensors() failed, the TFArenaSize is (probably) too small.");
        return nullptr; // Return null if tensor allocation fails
    }

    // Debugging to see the exact arena size
    if (_debug)
    {
        delay(1000);
        // Print the actual arena size
        size_t _used_size = _static_interpreter.arena_used_bytes();
        MicroPrintf("\r\nDEBUG:Tensor arena used %d bytes.\r\n", _used_size);
    }

    // Obtain pointers to the model's input and output tensors.
    TFLMinput = _static_interpreter.input(0);
    TFLMoutput = _static_interpreter.output(0);

    return &_static_interpreter; // Return the interpreter pointer
}

/*!
    @brief  Predict an output from the given input(s), set them using for example `TFLMinput->data.f[a] = b`
    @return True if the prediction was successful, False if an error occurred
*/
bool TFLMpredict()
{
    _previousMicros = micros();

    // Run inference, and report any error
    TfLiteStatus _invoke_status = TFLMinterpreter->Invoke();
    if (_invoke_status != kTfLiteOk)
    {
        MicroPrintf("Invoke failed."); // Log failure of the invoke operation
        return false;                  // Return false if invocation fails
    }

    if (_debug)
    {
        MicroPrintf("\r\nDEBUG:Prediction took %d microseconds.\r\n", micros() - _previousMicros);
    }

    return true; // Return true if prediction was successful
}
