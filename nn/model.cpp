#include <math.h>
#include "model.h"
#include "model_data.h"

inline float relu(float x) {
    return fmaxf(0.0f, x); 
}

float predict(float x) {
    // The activations of the first layer are small enough to store
    // on the stack (16 floats = 64 bytes).

    float h1[DENSE1_SIZE];

    // First dense layer. Since there is only one input neuron, we don't need 
    // to perform a full-blown matrix multiply.

    /*for (int i = 0; i < DENSE1_SIZE; ++i) {
        h1[i] = relu(x * W1(i) + b1(i));
    }*/

    for (int i = 0; i < DENSE1_SIZE; ++i) {
        float h1(0.0f);
        for (int j = 0; j < INPUT_SIZE; ++j) {
            h1[i] += x[j] * W1(i, j);
        }
        h1[i] = relu(h1[i] + b1(i));
    }

    // Second dense layer.

    float h2[DENSE1_SIZE];
    /*float y(0.0f);*/

    for (int i = 0; i < DENSE2_SIZE; ++i) {
        // Perform a dot product of the incoming activation vector with each 
        // row of the W2 matrix.
        /*float h2(0.0f);*/
        for (int j = 0; j < DENSE1_SIZE; ++j) {
            h2[i] += h1[j] * W2(i, j);
        }
        h2[i] = relu(h2[i] + b2(i));

        // We don't actually need to store the activations of the second layer. 
        // Since the last layer only has one neuron, we can immediately compute 
        // how much each activation contributes to the final layer.
        /*y += h2 * W3(i);*/
    }

    float h3[DENSE3_SIZE];

    for (int i = 0; i < DENSE3_SIZE; ++i) {
        // Perform a dot product of the incoming activation vector with each
        // row of the W2 matrix.
        /*float h2(0.0f);*/
        for (int j = 0; j < DENSE2_SIZE; ++j) {
            h3[i] += h2[j] * W3(i, j);
        }
        h3[i] = h2[i] + b2(i);

        // We don't actually need to store the activations of the second layer.
        // Since the last layer only has one neuron, we can immediately compute
        // how much each activation contributes to the final layer.
        /*y += h2 * W3(i);*/
    }

    // Final dense layer.
    return h3;
}
