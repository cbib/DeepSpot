import tensorflow.keras as keras
from tensorflow.keras import layers

import residual_blocks as blocks


def deepSpot(config, input_shape = (256, 256, 1)):
    
    # Define the input as a tensor with shape input_shape
    X_input = keras.Input(input_shape)
    
    # step 1 : 1st conv
    X_conv = layers.Conv2D(32, (3, 3), strides = (2, 2), padding="same", name = 'conv1')(X_input)
    X_conv = layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X_conv)
    X_conv = layers.Activation('relu')(X_conv)

    X_conv = layers.Conv2D(64, (3, 3), strides = (2, 2), padding="same", name = 'conv2')(X_conv)
    X_conv = layers.BatchNormalization(axis = 3, name = 'bn_conv2')(X_conv)
    X_conv = layers.Activation('relu')(X_conv)

    X_conv = layers.Conv2D(128, (3, 3), strides = (2, 2), padding="same", name = 'conv3')(X_conv)
    X_conv = layers.BatchNormalization(axis = 3, name = 'bn_conv3')(X_conv)
    X_conv = layers.Activation('relu')(X_conv)

    #Maxpool
    X_maxpool = layers.Conv2D(32, (3, 3), strides = (1, 1), padding="same", name = 'conv_maxpool1')(X_input)
    X_maxpool = layers.BatchNormalization(axis = 3, name = 'bn_maxpool1')(X_maxpool)
    X_maxpool = layers.Activation('relu')(X_maxpool)
    X_maxpool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(X_maxpool)

    X_maxpool = layers.Conv2D(64, (3, 3), strides = (1, 1), padding="same", name = 'conv_maxpool2')(X_maxpool)
    X_maxpool = layers.BatchNormalization(axis = 3, name = 'bn_maxpool2')(X_maxpool)
    X_maxpool = layers.Activation('relu')(X_maxpool)
    X_maxpool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(X_maxpool)

    X_maxpool = layers.Conv2D(128, (3, 3), strides = (1, 1), padding="same", name = 'conv_maxpool3')(X_maxpool)
    X_maxpool = layers.BatchNormalization(axis = 3, name = 'bn_maxpool3')(X_maxpool)
    X_maxpool = layers.Activation('relu')(X_maxpool)
    X_maxpool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(X_maxpool)

    #A trou
    X_atrou = layers.Conv2D(32, (3, 3), strides = (1, 1), padding="same", dilation_rate=(2, 2), name = 'conv_atrou1')(X_input)
    X_atrou = layers.BatchNormalization(axis = 3, name = 'bn_atrou1')(X_atrou)
    X_atrou = layers.Activation('relu')(X_atrou)
    X_atrou = layers.MaxPooling2D((3, 3), padding="same", strides=(2, 2))(X_atrou)

    X_atrou = layers.Conv2D(64, (3, 3), strides = (1, 1), padding="same", dilation_rate=(2, 2), name = 'conv_atrou2')(X_atrou)
    X_atrou = layers.BatchNormalization(axis = 3, name = 'bn_atrou2')(X_atrou)
    X_atrou = layers.Activation('relu')(X_atrou)
    X_atrou = layers.MaxPooling2D((3, 3), padding="same", strides=(2, 2))(X_atrou)

    X_atrou = layers.Conv2D(128, (3, 3), strides = (1, 1), padding="same", dilation_rate=(2, 2), name = 'conv_atrou3')(X_atrou)
    X_atrou = layers.BatchNormalization(axis = 3, name = 'bn_atrou3')(X_atrou)
    X_atrou = layers.Activation('relu')(X_atrou)
    X_atrou = layers.MaxPooling2D((3, 3), padding="same", strides=(2, 2))(X_atrou)

    X = layers.Concatenate()([X_conv, X_maxpool, X_atrou])


    F1 = config['conv_block4_filters']
    F2 = F1
    F3 = config['identity_block_filters']

    # step 4 : 6 blocs
    X = blocks.conv_block(X, k = 3, filters = [F1, F2, F3], stage = 3, block='a', s = 1, dropout_rate = config['dropout_rate'])

    F1 = F3
    F2 = F3

    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='b', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='c', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='d', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='e', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='f', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='g', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='h', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='i', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='j', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='k', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='l', dropout_rate =config['dropout_rate'])
    X = blocks.identity_block(X, k = 3, filters = [F1, F2, F3], stage=3, block='m', dropout_rate =config['dropout_rate'])

    #upconv
    X = blocks.conv_up_block(X, [256, 128, 128], stage=6, block='a', s = 2, dropout_rate =config['dropout_rate'])
    X = blocks.conv_up_block(X, [128, 64, 64], stage=6, block='b', s = 2, dropout_rate =config['dropout_rate'])
    X = blocks.conv_up_block(X, [64, 32, 32], stage=6, block='c', s = 2, dropout_rate =config['dropout_rate'])

    # output layer
    output = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(X)

    # Create model
    model = keras.Model(inputs = X_input, outputs = output, name='deepspot')

    return model
