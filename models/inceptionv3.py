import tensorflow as tf
from tensorflow.keras import layers, models

def InceptionV3(input_shape=(299, 299, 3), num_classes=1000):
    input_img = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input_img)
    x = layers.Conv2D(32, (3, 3), padding='valid')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(80, (1, 1), padding='valid')(x)
    x = layers.Conv2D(192, (3, 3), padding='valid')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Inception block 1
    x = inception_block(x, filters=[64, 96, 128, 16, 32, 32])
    x = inception_block(x, filters=[128, 128, 192, 32, 96, 64])

    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Inception block 2
    x = inception_block(x, filters=[192, 96, 208, 16, 48, 64])
    x = inception_block(x, filters=[160, 112, 224, 24, 64, 64])
    x = inception_block(x, filters=[128, 128, 256, 24, 64, 64])
    x = inception_block(x, filters=[112, 144, 288, 32, 64, 64])
    x = inception_block(x, filters=[256, 160, 320, 32, 128, 128])

    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Inception block 3
    x = inception_block(x, filters=[256, 160, 320, 32, 128, 128])
    x = inception_block(x, filters=[384, 192, 384, 48, 128, 128])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_img, outputs=output)
    
    return model

def inception_block(x, filters):
    f1, f3_r, f3, f5_r, f5, f_pool = filters
    
    conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(x)
    
    conv3 = layers.Conv2D(f3_r, (1, 1), padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(f3, (3, 3), padding='same', activation='relu')(conv3)
    
    conv5 = layers.Conv2D(f5_r, (1, 1), padding='same', activation='relu')(x)
    conv5 = layers.Conv2D(f5, (5, 5), padding='same', activation='relu')(conv5)
    
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool = layers.Conv2D(f_pool, (1, 1), padding='same', activation='relu')(pool)
    
    return layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
