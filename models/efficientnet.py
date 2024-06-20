import tensorflow as tf
from tensorflow.keras import layers, models

def mb_conv_block(inputs, filters, kernel_size, strides, expand_ratio):
    in_channels = inputs.shape[-1]
    x = layers.Conv2D(in_channels * expand_ratio, 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if strides == 1 and in_channels == filters:
        x = layers.Add()([inputs, x])
    
    return x

def EfficientNet(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = mb_conv_block(x, 16, 3, 1, 1)
    x = mb_conv_block(x, 24, 3, 2, 6)
    x = mb_conv_block(x, 24, 3, 1, 6)
    x = mb_conv_block(x, 40, 5, 2, 6)
    x = mb_conv_block(x, 40, 5, 1, 6)
    x = mb_conv_block(x, 80, 3, 2, 6)
    x = mb_conv_block(x, 80, 3, 1, 6)
    x = mb_conv_block(x, 80, 3, 1, 6)
    x = mb_conv_block(x, 112, 5, 1, 6)
    x = mb_conv_block(x, 112, 5, 1, 6)
    x = mb_conv_block(x, 192, 5, 2, 6)
    x = mb_conv_block(x, 192, 5, 1, 6)
    x = mb_conv_block(x, 192, 5, 1, 6)
    x = mb_conv_block(x, 192, 5, 1, 6)
    x = mb_conv_block(x, 320, 3, 1, 6)
    
    x = layers.Conv2D(1280, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='EfficientNet')
    
    return model
