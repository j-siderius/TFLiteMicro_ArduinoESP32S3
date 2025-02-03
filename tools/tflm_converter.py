# Modified from The TensorFlow project 2023
# Originally from https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver
# Modified by j-siderius <> https://github.com/j-siderius/

import re
import os
from mako import template
from tensorflow.lite.tools import visualize

_build_template = template.Template("""/*
    Modified from The TensorFlow project 2023
    Originally from https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver
    Modified by j-siderius <> https://github.com/j-siderius/
*/

// This file was generated based on ${model}.
// Include the following line in the setup of the main INO file to initialise the model: 
// `TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, ${tensor_arena_size}>(${model_name}, TFLMgetResolver);`

#pragma once

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

constexpr int TFLMnumberOperators = ${number_of_ops};

inline tflite::MicroMutableOpResolver<TFLMnumberOperators> TFLMgetResolver()
{
  tflite::MicroMutableOpResolver<TFLMnumberOperators> micro_op_resolver;

% for operator in operators:
  micro_op_resolver.${operator}();
% endfor

  return micro_op_resolver;
}

// Align the model structure to the architecture it is compiled for
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

unsigned int TFLMmodelLength = ${model_length};

const unsigned char ${model_name}[] DATA_ALIGN_ATTRIBUTE = {
    ${hex_array}
};""")


def GetModelOperatorsAndActivation(model_path):
    """Extracts a set of operators from a tflite model."""

    custom_op_found = False
    operators_and_activations = set()

    with open(model_path, 'rb') as f:
        data_bytes = bytearray(f.read())

    data = visualize.CreateDictFromFlatbuffer(data_bytes)

    for op_code in data["operator_codes"]:
        if op_code['custom_code'] is None:
            op_code["builtin_code"] = max(op_code["builtin_code"],
                                          op_code["deprecated_builtin_code"])
        else:
            custom_op_found = True
            operators_and_activations.add(
                visualize.NameListToString(op_code['custom_code']))

    for op_code in data["operator_codes"]:
        # Custom operator already added.
        if custom_op_found and visualize.BuiltinCodeToName(
                op_code['builtin_code']) == "CUSTOM":
            continue

        operators_and_activations.add(
            visualize.BuiltinCodeToName(op_code['builtin_code']))

    return operators_and_activations


def ParseTFLMOperatorString(word):
    """Converts a flatbuffer operator string to a format suitable for Micro
       Mutable Op Resolver. Example: CONV_2D --> AddConv2D."""

    # Edge case for AddDetectionPostprocess().
    # The custom code is TFLite_Detection_PostProcess.
    word = word.replace('TFLite', '')

    word_split = re.split('_|-', word)
    formated_op_string = ''
    for part in word_split:
        if len(part) > 1:
            if part[0].isalpha():
                formated_op_string += part[0].upper() + part[1:].lower()
            else:
                formated_op_string += part.upper()
        else:
            formated_op_string += part.upper()

    # Edge cases
    formated_op_string = formated_op_string.replace('Lstm', 'LSTM')
    formated_op_string = formated_op_string.replace('BatchMatmul', 'BatchMatMul')

    return 'Add' + formated_op_string


def VerifyTFLMOperatorList(op_list, header):
    """Make sure operators in list are not missing in header file ."""

    supported_op_list = []
    with open(header, 'r') as f:
        for l in f.readlines():
            if "TfLiteStatus Add" in l:
                op = l.strip().split(' ')[1].split('(')[0]
                supported_op_list.append(op)

    for op in op_list:
        if op not in supported_op_list:
            print(f'{op} not supported by TFLM')
            return False
        else:
            # print(f"{op} is supported by TFLM")
            pass

    return True


def GenerateTFLMHexModel(model_path):
    # Open the TFLite model file in binary mode and read its content into 'tflite_content'.
    with open(model_path, 'rb') as tflite_file:
        tflite_content = tflite_file.read()

    # Calculate the length of 'tflite_content' (i.e., the size of the TFLite model in bytes).
    array_length = len(tflite_content)

    # Split 'tflite_content' into chunks of 12 bytes each and convert each chunk to a hexadecimal string.
    # This is done so that the TFLite model can be represented as an array in C-compatible format.
    hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in
                 range(0, len(tflite_content), 12)]

    # Join the chunks of hexadecimal strings with newlines to format them neatly.
    hex_array = ',\n     '.join(hex_lines)

    return hex_array, array_length


def GenerateTFLMHeaderFile(build_template, model_ops, model_name, model_hex_array, model_length, output_dir):
    """Generates Micro Mutable Op Resolver code based on a template."""

    number_of_ops = len(model_ops)
    outfile = model_name + '_model.h'

    template_file_path = "./tflm_resolver.h.mako"
    # build_template = template.Template(filename=template_file_path)

    with open(output_dir + '/' + outfile, 'w') as file_obj:
        key_values_in_template = {
            'model': model_name + ".tflite",
            'number_of_ops': number_of_ops,
            'operators': model_ops,
            'model_length': model_length,
            'model_name': "TFLM_" + model_name + "_model",
            'hex_array': model_hex_array,
            'tensor_arena_size': (int(41240/5000)+1)*5000
        }
        file_obj.write(build_template.render(**key_values_in_template))

# -----------------------------------------------------------------------------------

def convert_tflite_to_tflm(tflite_path: str = "model.tflite") -> str:
    """
    Converts TFLite models into C-compatible header files for use in Arduino.
    Generates a corresponding TFLite Micro Operations resolver function.

    Parameters
    ----------
    tflite_path : str
        Path to the TFLite model file.
        Default path is 'model.tflite'.
        The name of this model will determine the final header file name.

    Returns
    -------
    model_path : str
        Path to the converted C-compatible header file

    Raises
    ------
    LookupError
        If the converter is missing the required base files
    ValueError
        If the provided tflite_path does not contain a TFLite model
    NotImplementedError
        If an operator from the TFLite model is not supported by the TFLite Micro kernel
    """

    # Get the current directory filepath
    path = os.path.dirname(os.path.abspath(__file__))

    # Check if all required files exist
    if not os.path.isfile(path + "/micro_mutable_op_resolver.h"):
        raise LookupError("The required base files (micro_mutable_op_resolver.h) "
                          "are not present.")

    # Check if file is a tflite model. If not, raise ValueError.
    if not tflite_path.endswith('.tflite'):
        raise ValueError("The provided file is (probably) not a TFLite model.")

    operator_list = []
    model_name = tflite_path.split('/')[-1].split('.')[0]
    print(model_name)

    operators = GetModelOperatorsAndActivation(tflite_path)

    for op in sorted(list(operators)):
        operator_list.append(ParseTFLMOperatorString(op))

    if not VerifyTFLMOperatorList(operator_list, path + '/micro_mutable_op_resolver.h'):
        raise NotImplementedError("Not all operations could be added to TFLM, aborting conversion.")

    hex_array, array_length = GenerateTFLMHexModel(tflite_path)

    GenerateTFLMHeaderFile(_build_template, operator_list, model_name, hex_array, array_length, "./")

    print(f"{model_name} has been converted and saved to {model_name + "_model.h"}")
    return model_name + "_model.h"


if __name__ == "__main__":
    convert_tflite_to_tflm('model.tflite')
