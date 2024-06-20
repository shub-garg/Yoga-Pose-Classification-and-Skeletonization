import tensorflow as tf
from tensorflow.keras import layers, models

def inception_resnet_block(x, scale, block_type, block_idx):
    if block_type == 'block35':
        branch_0 = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
        
        branch_1 = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
        branch_1 = layers.Conv2D(32, 3, padding='same', activation='relu')(branch_1)
        
        branch_2 = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
        branch_2 = layers.Conv2D(48, 3, padding='same', activation='relu')(branch_2)
        branch_2 = layers.Conv2D(64, 3, padding='same', activation='relu')(branch_2)
        
        branches = [branch_0, branch_1, branch_2]
    
    elif block_type == 'block17':
        branch_0 = layers.Conv2D(192, 1, padding='same', activation='relu')(x)
        
        branch_1 = layers.Conv2D(128, 1, padding='same', activation='relu')(x)
        branch_1 = layers.Conv2D(160, [1, 7], padding='same', activation='relu')(branch_1)
        branch_1 = layers.Conv2D(192, [7, 1], padding='same', activation='relu')(branch_1)
        
        branches = [branch_0, branch_1]
    
    elif block_type == 'block8':
        branch_0 = layers.Conv2D(192, 1, padding='same', activation='relu')(x)
        
        branch_1 = layers.Conv2D(192, 1, padding='same', activation='relu')(x)
        branch_1 = layers.Conv2D(224, [1, 3], padding='same', activation='relu')(branch_1)
        branch_1 = layers.Conv2D(256, [3, 1], padding='same', activation='relu')(branch_1)
        
        branches = [branch_0, branch_1]
    
    mixed = layers.Concatenate()(branches)
    up = layers.Conv2D(tf.keras.backend.int_shape(x)[-1], 1, activation='linear', padding='same')(mixed)
    up = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=tf.keras.backend.int_shape(x)[1:], arguments={'scale': scale})([x, up])
    x = layers.Activation('relu')(up)
    
    return x

def InceptionResNetV2(input_shape=(299, 299, 3), num_classes=1000):
    input_img = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input_img)
    x = layers.Conv2D(32, (3, 3), padding='valid')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(80, (1, 1), padding='valid')(x)
    x = layers.Conv2D(192, (3, 3), padding='valid')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Mixed 5b (Inception-A block)
    branch_0 = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(x)
    
    branch_1 = layers.Conv2D(48, (1, 1), padding='same', activation='relu')(x)
    branch_1 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(branch_1)
    
    branch_2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    branch_2 = layers.Conv2D(96, (3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2D(96, (3, 3), padding='same', activation='relu')(branch_2)
    
    branch_3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_3 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(branch_3)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2, branch_3])
    
    # 10x block35 (Inception-ResNet-A block)
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx)
    
    # Reduction-A block
    branch_0 = layers.Conv2D(384, (3, 3), strides=(2, 2), padding='valid', activation='relu')(x)
    
    branch_1 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    branch_1 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(branch_1)
    branch_1 = layers.Conv2D(384, (3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_1)
    
    branch_2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2])
    
    # 20x block17 (Inception-ResNet-B block)
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx)
    
    # Reduction-B block
    branch_0 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    branch_0 = layers.Conv2D(384, (3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_0)
    
    branch_1 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    branch_1 = layers.Conv2D(288, (3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_1)
    
    branch_2 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    branch_2 = layers.Conv2D(288, (3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2D(320, (3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_2)
    
    branch_3 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2, branch_3])
    
    # 10x block8 (Inception-ResNet-C block)
    for block_idx in range(1, 10):
        x = inception_resnet_block(x, scale=0.2, block_type='block8', block_idx=block_idx)
    
    x = inception_resnet_block(x, scale=1.0, block_type='block8', block_idx=10)
    
    x = layers.Conv2D(1536, 1, activation='relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.8)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(input_img, output, name='InceptionResNetV2')
    
    return model
