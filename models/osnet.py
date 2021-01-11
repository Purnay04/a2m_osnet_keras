from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils


if backend.image_data_format() != 'channels_last':
    raise RuntimeError('only works for channel_last')

def conv2d_bn(x,
              filters,
              kernel_size=(3, 3),
              padding='same',
              strides=(1, 1),
              activation='relu'):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size of the convolution
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    bn_axis = 3
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    if activation is not None:
        x = layers.Activation(activation)(x)

    return x 
def light_conv3x3_bn(x,
                     filters):
    """Utility function of lightweight 3x3 convolution.
    lightweight 3x3 = 1x1 (linear) + dw 3x3 (nonlinear).
    # Arguments
        x: input tensor.
        filters: number of output filters.
    # Returns
        Output tensor.
    """
    bn_axis = 3 
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    return x

def get_aggregation_gate(in_filters, reduction=16):
    """Get the "aggregation gate (AG)" op.
    # Arguments
        reduction: channel reduction for the hidden layer.
    # Returns
        The AG op (a models.Sequential module).
    """
    gate = tf.keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(in_filters // reduction, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(in_filters),
        layers.Activation('sigmoid'),
        layers.Reshape((1, 1, -1)) # reshape as (H, W, C)
    ])

    return gate

def os_bottleneck(x, out_filters, bottleneck_reduction=4):
    """Utility function to implement the OSNet bottleneck module.
    # Arguments
        x: input tensor.
        out_filters: number of output filters.
    # Returns
        Output tensor after applying the OSNet bottleneck.
    """
    in_filters = x.shape[-1].value
    mid_filters = out_filters // bottleneck_reduction
    identity = x 
    x1 = conv2d_bn(x, mid_filters, kernel_size=(1, 1))

    branch1 = light_conv3x3_bn(x1, mid_filters)

    branch2 = light_conv3x3_bn(x1, mid_filters)
    branch2 = light_conv3x3_bn(branch2, mid_filters)

    branch3 = light_conv3x3_bn(x1, mid_filters)
    branch3 = light_conv3x3_bn(branch3, mid_filters)
    branch3 = light_conv3x3_bn(branch3, mid_filters)

    branch4 = light_conv3x3_bn(x1, mid_filters)
    branch4 = light_conv3x3_bn(branch4, mid_filters)
    branch4 = light_conv3x3_bn(branch4, mid_filters)
    branch4 = light_conv3x3_bn(branch4, mid_filters)

    gate = get_aggregation_gate(mid_filters)

    x2 = layers.Add()([
        layers.Multiply()([branch1, gate(branch1)]),
        layers.Multiply()([branch2, gate(branch2)]),
        layers.Multiply()([branch3, gate(branch3)]),
        layers.Multiply()([branch4, gate(branch4)])
    ])

    x3 = conv2d_bn(x2, out_filters, kernel_size=(1, 1), activation=None)

    if in_filters != out_filters:
        identity = conv2d_bn(identity, out_filters, kernel_size=(1, 1), activation=None)
    
    out = layers.Add()([identity, x3]) # residual connection, out = x3 + identity in Pytorch
    out = layers.Activation('relu')(out)

    return out 

def OSNet(include_top=False,
          weights=None,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          **kwargs):
    """Instantiates the OSNet (256x128) architecture.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: must be None.
        input_tensor: Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: input tensor shape, which is used to create an
            input tensor if `input_tensor` is not specified.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `avg` means that global average pooling will be applied
                to the output of the last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    if weights is not None:
        raise ValueError('weights is not currently supported')
    if input_tensor is None:
        if input_shape is None:
            raise ValueError('neither input_tensor nor input_shape is given')
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv2d_bn(img_input, 64, (7, 7), strides=(2, 2)) # conv1: 128x64x64
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x) # 1p: 64x32x64

    x = os_bottleneck(x, 256)                                           # 2a: 64x32x256
    x = os_bottleneck(x, 256)                                           # 2b: 64x32x256
    x = conv2d_bn(x, 256, (1, 1))                                       # 2t: 64x32x256
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)    # 2p: 32x16x256

    x = os_bottleneck(x, 384)                                           # 3a: 16x8x384
    x = os_bottleneck(x, 384)                                           # 3b: 16x8x384
    x = conv2d_bn(x, 384, (1, 1))                                       # 3t: 16x8x384
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)    # 3p: 8x4x384

    x = os_bottleneck(x, 512)                                           # 3a: 8x4x512
    x = os_bottleneck(x, 512)                                           # 3b: 8x4x512
    x = conv2d_bn(x, 512, (1, 1))                                       # 3t: 8x4x512

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(classes, activation='softmax')(x)
    
    #
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = models.Model(inputs, x)

    return model
