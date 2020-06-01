#!/usr/bin/python3

"""custom_keras_layers.py

Implementation of various custom Keras Layers 
"""

__author__ = "Bas Straathof"


import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Layer, \
        multiply, Reshape


def squeeze_excite_block(X, mask=None):
    """Squeeze and excitation block
    
    (Hu et al. 2019 - Squeeze-and-Excitation Networks)
    
    Args:
        X (tf.Tensor): Input tensor
        mask (tf.Tensor): Mask to mask out padded zeros in the
                  GlobalAveragePooling1D layer
        
    Returns:
        X (tf.Tensor): Transformed input tensor 
    """
    filters = X.shape[-1]
    r = 16

    # Squeeze
    se = GlobalAveragePooling1D()(X, mask)
    se = Reshape((1, filters))(se)

    # Excite
    se = Dense(filters // r,  activation='relu',
            kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
            use_bias=False)(se)

    # Transform the input
    X = multiply([X, se])

    return X


class Slice(Layer):
    """Take a channel (i.e. feature-mask combination) slice of a 3D tensor"""
    def __init__(self, feature, mask, **kwargs):
        self.slice = [feature, mask]
        super(Slice, self).__init__(**kwargs)

    def call(self, X, mask=None):
        # Permute the tensor
        X = tf.transpose(X, perm=(2, 0, 1))

        # Gather the feature and corresponding mask
        X = tf.gather(X, self.slice)

        # Re-permute the tensor
        X = tf.transpose(X, perm=(1, 2, 0))

        return X

    # Layer must override get_config
    def get_config(self):
        return {'slice': self.slice}
    
    
class ApplyMask(Layer):
    """Apply a previously calculated mask to the output of a  differentlayer"""
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(ApplyMask, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # Make sure not to pass the mask to the next layer
        return None

    def call(self, x, mask=None):
        if mask is not None:
            # Turn boolean mask to floats (batch_size, time_steps)
            mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
            # Repeat the mask along the lsat dimension of the input
            # (batch_size, x_dim, time_steps)
            mask = tf.keras.backend.repeat(mask, x.shape[-1])
            # Permute the mask ask (batch_size, time_steps, x_dim)
            mask = tf.transpose(mask, [0, 2, 1])
            # Apply the mask to the input
            x = x * mask

        return x

    def get_output_shape_for(self, input_shape):
        # Remove the temporal dimension (i.e. time_steps)
        return input_shape[0], input_shape[2]

