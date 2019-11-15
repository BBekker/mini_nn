/*
 * activation.h
 *
 *  Created on: 13 Nov 2019
 *      Author: Bernard
 */

#ifndef NN_LAYERS_MINI_NN_H_
#define NN_LAYERS_MINI_NN_H_

#include "math.h"

typedef void (*layer_func_t)(const float *input,  const unsigned int input_len, const unsigned int output_len, const float *weights, float *output);
typedef void (*activation_func_t)(const float *vector, const unsigned int length, float *output);

typedef struct
{
    float scale;
    float offset;
} mini_nn_scaler_weights_t;

typedef struct
{
    layer_func_t layer_function;
    activation_func_t activation_function;
    unsigned int input_size;
    unsigned int output_size;
    union
    {
        float *weights;
        mini_nn_scaler_weights_t *scaler_weights;
    };
} mini_nn_layer_t;

void apply_layer(mini_nn_layer_t* layer, const float *input, float *output);


/* layers */
void fc_layer(const float *input,  const unsigned int input_len, const unsigned int output_len, const float *weights, float *output);
void norm_layer(const float *input,  const unsigned int input_len, const unsigned int output_len, const float *weights, float *output);

/* Activation functions */
void reLu(const float *vector, const unsigned int length, float *output);
void softmax(const float *vector, const unsigned int length, float *output);


#endif /* NN_LAYERS_MINI_NN_H_ */
