#include "TFLiteMicro_ArduinoESP32S3.h"
#include "trained_lstm_model.h"
#include "digits.h"

void setup()
{
    Serial.begin(115200); // Initialize serial communication at 115200 baud rate

    // Set up the TensorFlow Lite Micro interpreter with the trained LSTM model
    TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 45000>(TFLM_trained_lstm_model, TFLMgetResolver);

    // Check if the model was set up correctly
    if (!TFLMinterpreter)
    {
        MicroPrintf("The model was not setup correctly."); // Log error if setup fails
        for (;;)
        {
            // Infinite loop to halt execution if model setup fails
        }
    }
}

void loop()
{
    // Generate a random digit between 0 and 9
    int random_digit = random(0, 10); 

    // Retrieve the corresponding digit data from the digits array
    float(*digit)[28][28] = digits[random_digit]; 

    // Fill the input tensor with the digit data (28x28 pixel values)
    for (int row = 0; row < 28; row++)
    {
        for (int column = 0; column < 28; column++)
        {
            int nr = (column * 28) + row; // Calculate the index for the input tensor
            TFLMinput->data.f[nr] = (*digit)[column][row]; // Assign pixel value to input tensor
        }
    }

    TFLMpredict(); // Perform inference using the model

    float y_predicted[10]; // Array to store the prediction results

    // Show the number to be predicted
    MicroPrintf("Predicting number %d", random_digit);
    // Log the header for the prediction results
    Serial.println("digit:\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9");
    Serial.print("proba:\t"); // Log the header for probabilities

    // Print the prediction results for each digit
    for (int i = 0; i < 10; i++)
    {
        y_predicted[i] = TFLMoutput->data.f[i]; // Store the predicted probabilities
        Serial.print(y_predicted[i], 5); // Print the probability with 5 decimal places
        Serial.print("\t"); // Tab space for formatting
    }
    Serial.println();  // newline for correct display in serial

    delay(1000); // Delay for 1 second before the next loop iteration
}
