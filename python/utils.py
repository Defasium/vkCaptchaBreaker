#-*- coding:utf-8 -*-
#'''
# Created on 18-8-14 下午4:39
#
# @Author: Greg Gao(laygin)
#'''
import os
import numpy as np
from tensorflow.keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense
from tensorflow.keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1, act="relu", pre="", batchnorm=True):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = '{}stage{}/block{}'.format(pre, stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    if batchnorm:
        x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation(act, name='{}/{}_1x1conv_1'.format(prefix, act))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    if batchnorm:
        x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    if batchnorm:
        x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation(act, name='{}/{}_1x1conv_2'.format(prefix, act))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        if batchnorm:
            s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        if batchnorm:
            s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation(act, name='{}/{}_1x1conv_3'.format(prefix, act))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1, act="relu", first_stride=2, prefix="", batchnorm=True):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=first_stride,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1, act=act, pre=prefix, batchnorm=batchnorm)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1], strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i), act=act, pre=prefix, batchnorm=batchnorm)

    return x