#!/usr/bin/python3.6																															# -*- coding: utf-8 -*-	

from tensorflow.keras import layers


def identity_block(x, k, filters, stage, block, dropout_rate=0.3):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    # create shortcut
    X_shortcut = x

    # main path conv1
    x = layers.BatchNormalization(axis=3, name=bn_name + '2a')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2a')(x)

    # main path conv2
    x = layers.BatchNormalization(axis=3, name=bn_name + '2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=F2, kernel_size=(k, k), strides=(1, 1), padding='same', name=conv_name + '2b')(x)

    # main path conv3
    x = layers.BatchNormalization(axis=3, name=bn_name + '2c')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2c')(x)

    x = layers.SpatialDropout2D(dropout_rate)(x)

    # main path, concat and layers.Activation
    x = layers.Add()([x, X_shortcut])

    return x


def conv_block(X, k, filters, stage, block, s=2, dropout_rate=0.1):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    # create shortcut
    X_shortcut = X

    ##### MAIN PATH
    # main path conv1
    X = layers.BatchNormalization(axis=3, name=bn_name + '2a')(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name + '2a')(X)

    # main path conv2
    X = layers.BatchNormalization(axis=3, name=bn_name + '2b')(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(F2, (k, k), strides=(1, 1), padding='same', name=conv_name + '2b')(X)

    # main path conv3
    X = layers.BatchNormalization(axis=3, name=bn_name + '2c')(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name + '2c')(X)

    X = layers.SpatialDropout2D(dropout_rate)(X)

    ##### SHORTCUT PATH
    X_shortcut = layers.Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name + '1')(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3, name=bn_name + '1')(X_shortcut)

    # concat
    X = layers.Add()([X, X_shortcut])

    return X


def conv_up_block(X, filters, stage, block, s, dropout_rate=0.1):
    conv_name = 'up_res' + str(stage) + block + '_branch'
    bn_name = 'up_bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2DTranspose(F1, kernel_size=(3, 3), strides=(s, s), padding='same', kernel_initializer='he_normal',
                               name=conv_name + '2a')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2DTranspose(F2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                               name=conv_name + '2b')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2DTranspose(F3, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                               name=conv_name + '2c')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name + '2c')(X)

    X = layers.SpatialDropout2D(dropout_rate)(X)

    #### SHORTCUT PATH
    X_shortcut = layers.Conv2DTranspose(F2, kernel_size=(3, 3), strides=(s, s), padding='same', name=conv_name + '1')(
        X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3, name=bn_name + '1')(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    return X
