#!/usr/bin/python3
'''Keras implementation of Google's Inception Architectures.
Examples:
    To use this module, import it in your script or jupyter notebook with the following command:
        # from recognition_microarchitectures import MiniGoogLeNet

.. _VKontakte captcha bypass with pseudoCRNN model running as chrome extension:
   https://github.com/Defasium/VKCaptchaBreaker
'''


from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Activation, Dropout, Dense
from tensorflow.python.keras.layers import Flatten, Input, concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K


class MiniGoogLeNet:
    '''Class implementing Inception architecture in Keras.
    Attributes:
        None

    '''
    @staticmethod
    def conv_module(layer, K, kX, kY, stride=(1, 1), chan_dim=-1, padding='same', trainable=True):
        '''Inception Conv Layer
        Args:
            layer (tf.keras.Layer): input layer.
            K (int): number of filters,
            kX (int): kernel size at X dimension.
            kY (int): kernel size at Y dimension.
            stride (tuple, default is (1, 1)): strides for convolution
            chan_dim (int, default is -1): channels format, either last (-1) or first.
            padding (str, default='same'): padding for convolution
            trainable (bool, default is True): if False then BatchNormalization
                won't update weights

        Returns:
            new_layer (tf.keras.Layer): new Layer after Conv+BN+Act

        '''
        layer = Conv2D(K, (kX, kY), strides=stride, padding=padding,
                       kernel_initializer='he_normal')(layer)
        layer = BatchNormalization(axis=chan_dim, trainable=trainable)(layer)
        layer = Activation('relu')(layer)

        # return the block
        return layer

    @staticmethod
    def conv_module_fact(layer, K=64, kX=3, kY=3, stride=(1, 1), chan_dim=-1,
                         padding='same', trainable=True):
        '''Factorized Inception Conv Block with fewer parameters.
        Args:
            layer (tf.keras.Layer): input layer.
            K (int): number of filters,
            kX (int): kernel size at X dimension.
            kY (int): kernel size at Y dimension.
            stride (tuple, default is (1, 1)): strides for convolution
            chan_dim (int, default is -1): channels format, either last (-1) or first.
            padding (str, default='same'): padding for convolution
            trainable (bool, default is True): if False, then BatchNormalization
                won't update weights

        Returns:
            new_layer (tf.keras.Layer): new Layer after Conv+BN+Act

        '''
        layer = Conv2D(K, (1, kY), strides=(1, stride[-1]), padding=padding,
                       kernel_initializer='he_normal')(layer)
        layer = Conv2D(K, (kX, 1), strides=(stride[0], 1), padding=padding,
                       kernel_initializer='he_normal')(layer)
        layer = BatchNormalization(axis=chan_dim, trainable=trainable)(layer)
        layer = Activation('relu')(layer)

        # return the block
        return layer

    @staticmethod
    def inceptionv1_module(layer, numK1x1, numK3x3, chan_dim=-1, trainable=True):
        '''Factorized InceptionV2 Conv Module.
        Args:
            layer (tf.keras.Layer): input layer.
            numK1x1 (int): number of filters in 1x1 convolution,
            numK3x3 (int): number of filters in 3x3 factorized convolution.
            chan_dim (int, default is -1): channels format, either last (-1) or first.
            trainable (bool, default is True): if False, then BatchNormalization
                won't update weights

        Returns:
            new_layer (tf.keras.Layer): new Layer after Factorized InceptionV2 Conv Module

        '''
        conv_1x1 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1),
                                             chan_dim, trainable=trainable)
        conv_3x3 = MiniGoogLeNet.conv_module_fact(layer, numK3x3, 3, 3, (1, 1),
                                                  chan_dim, trainable=trainable)
        layer = concatenate([conv_1x1, conv_3x3], axis=chan_dim)

        # return the block
        return layer

    @staticmethod
    def downsample_module(layer, K, chan_dim=-1):
        '''Downsample InceptionV2 Module.
        Args:
            layer (tf.keras.Layer): input layer.
            K (int): number of filters in 3x3 strided factorized convolution,
            chan_dim (int, default is -1): channels format, either last (-1) or first.

        Returns:
            new_layer (tf.keras.Layer): new Layer

        '''
        conv_3x3 = MiniGoogLeNet.conv_module_fact(layer, K, 3, 3, (2, 2),
                                                  chan_dim, padding='valid')
        pool = MaxPooling2D((3, 3), strides=(2, 2))(layer)
        layer = concatenate([conv_3x3, pool], axis=chan_dim)

        # return the block
        return layer

    @staticmethod
    def inceptionv3_moduleA(layer, numK1x1, numK3x3, chan_dim):
        '''InceptionV3 Module A.
        Args:
            layer (tf.keras.Layer): input layer.
            numK1x1 (int): number of filters in 1x1 convolution,
            numK3x3 (int): number of filters in 3x3 convolution,
            chan_dim (int, default is -1): channels format, either last (-1) or first.

        Returns:
            new_layer (tf.keras.Layer): new Layer after InceptionV3 Module A

        '''
        conv_1x1 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)

        pool = MaxPooling2D((3, 3), strides=(2, 2))(layer)
        pool_conv_1x1 = MiniGoogLeNet.conv_module(pool, numK1x1, 1, 1, (1, 1), chan_dim)

        conv_1x1_2 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_1x1_3x3 = MiniGoogLeNet.conv_module(conv_1x1_2, numK3x3, 3, 3, (1, 1), chan_dim)

        conv_1x1_3 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_3x3 = MiniGoogLeNet.conv_module(conv_1x1_3, numK3x3, 3, 3, (1, 1), chan_dim)
        conv_3x3 = MiniGoogLeNet.conv_module(conv_3x3, numK3x3, 3, 3, (1, 1), chan_dim)
        layer = concatenate([conv_1x1, pool_conv_1x1, conv_1x1_3x3, conv_3x3], axis=chan_dim)

        # return the block
        return layer

    @staticmethod
    def inceptionv3_moduleB(layer, numK1x1, numK7x7, chan_dim):
        conv_1x1 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)

        pool = MaxPooling2D((3, 3), strides=(2, 2))(layer)
        pool_conv_1x1 = MiniGoogLeNet.conv_module(pool, numK1x1, 1, 1, (1, 1), chan_dim)

        conv_1x1_2 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_1x1_7x7 = MiniGoogLeNet.conv_module_fact(conv_1x1_2, numK7x7, 7, 7, (1, 1), chan_dim)

        conv_1x1_3 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_7x7 = MiniGoogLeNet.conv_module_fact(conv_1x1_3, numK7x7, 3, 3, (1, 1), chan_dim)
        conv_7x7_x2 = MiniGoogLeNet.conv_module_fact(conv_7x7, numK7x7, 3, 3, (1, 1), chan_dim)
        layer = concatenate([conv_1x1, pool_conv_1x1, conv_1x1_7x7, conv_7x7_x2], axis=chan_dim)

        # return the block
        return layer

    @staticmethod
    def inceptionv3_moduleC(layer, numK1x1, numK3x3, chan_dim):
        '''InceptionV3 Module C.
        Args:
            layer (tf.keras.Layer): input layer.
            numK1x1 (int): number of filters in 1x1 convolution,
            numK3x3 (int): number of filters in 3x3 convolution,
            chan_dim (int, default is -1): channels format, either last (-1) or first.

        Returns:
            new_layer (tf.keras.Layer): new Layer after InceptionV3 Module C

        '''
        conv_1x1 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)

        pool = MaxPooling2D((3, 3), strides=(2, 2))(layer)
        pool_conv_1x1 = MiniGoogLeNet.conv_module(pool, numK1x1, 1, 1, (1, 1), chan_dim)

        conv_1x1_2 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_1x1_1x3 = MiniGoogLeNet.conv_module(conv_1x1_2, numK3x3, 1, 3, (1, 1), chan_dim)
        conv_1x1_3x1 = MiniGoogLeNet.conv_module(conv_1x1_2, numK3x3, 3, 1, (1, 1), chan_dim)

        conv_1x1_3 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_1x1_3x3 = MiniGoogLeNet.conv_module(conv_1x1_3, numK3x3, 3, 3, (1, 1), chan_dim)
        conv_1x1_3x3_1x3 = MiniGoogLeNet.conv_module(conv_1x1_3x3, numK3x3, 1, 3, (1, 1), chan_dim)
        conv_1x1_3x3_3x1 = MiniGoogLeNet.conv_module(conv_1x1_3x3, numK3x3, 3, 1, (1, 1), chan_dim)

        layer = concatenate([conv_1x1, pool_conv_1x1, conv_1x1_1x3,
                             conv_1x1_3x1, conv_1x1_3x3_1x3, conv_1x1_3x3_3x1],
                            axis=chan_dim)

        # return the block
        return layer

    @staticmethod
    def efficient_gridsize_reduction(layer, numK1x1, numK3x3, chan_dim):
        '''Efficient GridSize Reduction Block.
        Args:
            layer (tf.keras.Layer): input layer.
            numK1x1 (int): number of filters in 1x1 convolution,
            numK3x3 (int): number of filters in 3x3 convolution,
            chan_dim (int, default is -1): channels format, either last (-1) or first.

        Returns:
            new_layer (tf.keras.Layer): new Layer after Efficient GridSize Reduction Block

        '''
        pool = MaxPooling2D((3, 3), strides=(2, 2))(layer)

        conv_1x1 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_1x1_3x3 = MiniGoogLeNet.conv_module(conv_1x1, numK3x3, 3, 3, (2, 2), chan_dim)

        conv_1x1_2 = MiniGoogLeNet.conv_module(layer, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_1x1_3x3_2 = MiniGoogLeNet.conv_module(conv_1x1_2, numK3x3, 3, 3, (1, 1), chan_dim)
        conv_1x1_3x3_2 = MiniGoogLeNet.conv_module(conv_1x1_3x3_2, numK3x3, 3, 3, (2, 2), chan_dim)

        layer = concatenate([pool, conv_1x1_3x3, conv_1x1_3x3_2], axis=chan_dim)

        # return the block
        return layer

    @staticmethod
    def build(width, height, depth, classes):
        '''Builds InceptionV2
        Args:
            width (int): width of input image.
            height (int): height of input image.
            depth (int): depth of input image.
            classes (int): number of classes for classification.

        Returns:
            Model (tf.keras.models.Model): InceptionV2 Model

        '''
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        inputs = Input(shape=input_shape)
        inner = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        inner = MiniGoogLeNet.inceptionv1_module(inner, 32, 32, chan_dim)
        inner = MiniGoogLeNet.inceptionv1_module(inner, 32, 48, chan_dim)
        inner = MiniGoogLeNet.downsample_module(inner, 80, chan_dim)

        # four Inception modules followed by a downsample modeule
        inner = MiniGoogLeNet.inceptionv1_module(inner, 112, 48, chan_dim)
        inner = MiniGoogLeNet.inceptionv1_module(inner, 96, 64, chan_dim)
        inner = MiniGoogLeNet.inceptionv1_module(inner, 80, 80, chan_dim)
        inner = MiniGoogLeNet.inceptionv1_module(inner, 48, 96, chan_dim)
        inner = MiniGoogLeNet.downsample_module(inner, 96, chan_dim)

        # two Inception modules followed by globalPool and dropout
        inner = MiniGoogLeNet.inceptionv1_module(inner, 176, 160, chan_dim)
        inner = MiniGoogLeNet.inceptionv1_module(inner, 176, 160, chan_dim)
        inner = AveragePooling2D((7, 7))(inner)
        inner = Dropout(0.5)(inner)

        # softmax classifier
        inner = Flatten()(inner)
        inner = Dense(classes)(inner)
        inner = Activation('softmax')(inner)

        # create the model
        model = Model(inputs, inner, name='googlenet')

        # return the constructed network architecture
        return model

    @staticmethod
    def build_v3(width, height, depth, classes):
        '''Builds InceptionV3
        https://miro.medium.com/max/700/1*gqKM5V-uo2sMFFPDS84yJw.png
        Args:
            width (int): width of input image.
            height (int): height of input image.
            depth (int): depth of input image.
            classes (int): number of classes for classification.

        Returns:
            Model (tf.keras.models.Model): InceptionV2 Model

        '''
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        # five Inception modules followed by a grid size reduction module
        inputs = Input(shape=input_shape)
        inner = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        inner = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        inner = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        inner = MiniGoogLeNet.downsample_module(inner, 80, chan_dim)
        inner = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        inner = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        inner = MiniGoogLeNet.downsample_module(inner, 80, chan_dim)

        # five Inception modules A followed by a grid size reduction module
        inner = MiniGoogLeNet.inceptionv3_moduleA(inner, 112, 48, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleA(inner, 96, 64, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleA(inner, 80, 80, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleA(inner, 64, 96, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleA(inner, 48, 112, chan_dim)
        inner = MiniGoogLeNet.efficient_gridsize_reduction(inner, 48, 48, chan_dim)

        # four Inception modules B followed by a grid size reduction module
        inner = MiniGoogLeNet.inceptionv3_moduleB(inner, 112, 48, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleB(inner, 96, 64, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleB(inner, 80, 80, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleB(inner, 48, 96, chan_dim)
        inner = MiniGoogLeNet.efficient_gridsize_reduction(inner, 48, 48, chan_dim)

        # two Inception modules C followed by globalPool and dropout
        inner = MiniGoogLeNet.inceptionv3_moduleC(inner, 176, 160, chan_dim)
        inner = MiniGoogLeNet.inceptionv3_moduleC(inner, 176, 160, chan_dim)
        inner = AveragePooling2D((7, 7))(inner)
        inner = Dropout(0.5)(inner)

        # softmax classifier
        inner = Flatten()(inner)
        inner = Dense(classes)(inner)
        inner = Activation('softmax')(inner)

        # create the model
        model = Model(inputs, inner, name='googlenetv3')

        # return the constructed network architecture
        return model
