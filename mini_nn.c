/*
 * activation.h
 *
 *  Created on: 13 Nov 2019
 *      Author: Bernard
 */

#include "mini_nn.h"
#include <ti/dsplib/dsplib.h>

#define NULL 0


void apply_layer(mini_nn_layer_t* layer, const float *input, float *output)
{
    if(layer->layer_function != NULL)
    {
        layer->layer_function(input, layer->input_size, layer->output_size, layer->weights, output);
    }

    if(layer->activation_function != NULL)
    {
        //Apply activation to all elements in output in place
        layer->activation_function(output, layer->output_size, output);
    }
}


/* layers */

void fc_layer(const float *input,  const unsigned int input_len, const unsigned int output_len, const float *weights, float *output)
{
    DSPF_sp_mat_mul(weights, output_len, input_len, input, 1, output);
}

void norm_layer(const float *input,  const unsigned int input_len, const unsigned int output_len, const float *weights, float *output)
{
    //Weights are structured as: [scale, offset][scale,offset]...
    mini_nn_scaler_weights_t *scaler_weights = (mini_nn_scaler_weights_t*) weights;

    for( int i = 0; i < input_len; i++)
    {
        output[i] = (input[i] - scaler_weights[i].offset) * scaler_weights[i].scale;
    }
}

/* Activation functions */

void reLu(const float *vector, const unsigned int length, float *output)
{
    for(unsigned int i = 0; i < length; i++)
    {
        output[i] = (vector[i] < 0) ? 0 : vector[i];
    }
}

void softmax(const float *vector, const unsigned int length, float *output)
{
    float sum = 0;
    for(unsigned int i = 0; i< length; i++)
    {
        sum += vector[i];
    }

    for(unsigned int i = 0; i< length; i++)
    {
        output[i] = vector[i] / sum;
    }
}

