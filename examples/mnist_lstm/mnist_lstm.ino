#include "TFLiteMicro_ArduinoESP32S3.h"
#include "trained_lstm_model.h"
#include "digits.h"

void setup()
{
    Serial.begin(115200);

    TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, 10000>(TFLM_trained_lstm_model, TFLMgetResolver, true);

    // Check if the model was set up correctly
    if (!TFLMinterpreter)
    {
        MicroPrintf("The model was not setup correctly.");
        for (;;)
        {
        }
    }
}

void loop()
{
    int random_digit = random(0, 10); // Generate a random digit

    Serial.println("Predicting number %d", random_digit);

    float(*digit)[28][28] = digits[random_digit]; // Get the corresponding digit data

    // Fill the input tensor with the digit data
    for (int row = 0; row < 28; row++)
    {
        for (int column = 0; column < 28; column++)
        {
            int nr = (column * 28) + row;
            TFLMinput->data.f[nr] = (*digit)[column][row];
        }
    }

    unsigned long startMicros = micros(); // Start timing

    TFLMpredict();

    // Print the time taken for inference
    Serial.println("Prediction took %d micros", micros() - startMicros);

    float y_predicted[10]; // Array to store the prediction results

    Serial.println("digit:\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9");
    Serial.print("proba:\t");

    // Print the prediction results
    for (int i = 0; i < 10; i++)
    {
        y_predicted[i] = TFLMoutput->data.f[i];
        Serial.print(y_predicted[i], 5);
        Serial.print("\t");
    }

    delay(1000); // Delay before the next loop
}