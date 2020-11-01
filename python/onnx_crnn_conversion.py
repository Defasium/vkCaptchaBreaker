#!/usr/bin/python3
'''Convertion from Keras to ONNX models
In this script the CRNN architecture can be converted into two ONNX models.
The first model is our trained Keras pseudoCRNN model, while the second one
is the Greedy-decoding algorithm of Connectionist Temporal Classification
via best-path finding in tensorgraph.

Two models is the tradeoff of troubles appearing with making one end-to-end model:
ONNX.js doesn't support some vital Operations, e.g. Cast (float -> int conversion)
or NonZero.
You can see the list of supported operators here:
	https://github.com/microsoft/onnxjs/blob/master/docs/operators.md

That's why to mitigate this problem, end-to-end model is divided into two,
and intermediate results are converted to Float32 format outside of the sessions
(in the JS).

Examples:
    To use this script try the following command:
        # python onnx_crnn_conversion.py --model_name='trained_crnn.h5'
	This will convert trained_crnn keras model into captcha.onnx and captcha_ctc.onnx models
	To specify name of the output models use the --onnxmodel_name option.

.. _VKontakte captcha bypass with pseudoCRNN model running as chrome extension:
   https://github.com/Defasium/VKCaptchaBreaker
'''


import argparse
import os

import tensorflow as tf

os.environ['TF_KERAS'] = '1'

import onnxmltools
from crnn import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Keras pseudoCRNN to ONNX models.')
    parser.add_argument('--model_name', type=str, action='store',
                        help='path to the Keras pseudoCRNN model')
    parser.add_argument('--onnxmodel_name', type=str, default='captcha_model', action='store',
                        help='name of the onnx models')
    parser.add_argument('--num_classes', type=int, default=23, action='store',
                        help='number of unique characters + 2')
    args = parser.parse_args()

    # construct small pseudoCRNN model and load weights
    model = get_model(training=False, onnx=True, input_shape=(64, 128, 4),
                      num_classes=args.num_classes, small=True, teacher=False)
    model.load_weights(args.model_name)

    # construct ctc greedy Decoder algorithm as Keras model
    inputs = tf.keras.layers.Input((30, ), dtype=tf.float32)
    y_pred = tf.keras.layers.Lambda(lambda x: args.num_classes - 1 - x, name='shift')(inputs)
    y_pred = tf.keras.layers.Reshape(target_shape=((30, 1, 1)), name='reshape_ctc')(y_pred)
    y_pred = tf.keras.layers.ZeroPadding2D(((1, 0), (0, 0)))(y_pred)
    y_pred = tf.keras.layers.Reshape(target_shape=((31,)), name='reshape_ctc2')(y_pred)
    y_pred = tf.keras.layers.Lambda(lambda x: (tf.abs(x[:, 1:]-x[:, :-1])*tf.abs(x[:, 1:]),
                                               args.num_classes - 1 - x[:, 1:]),
                                    name='filter')(y_pred)
    ctcmodel = tf.keras.models.Model(inputs, y_pred)

    # convert to onnx, please note that currently only 9 version is supported by ONNX.js
    onnx_ctcmodel = onnxmltools.convert_keras(ctcmodel, target_opset=9)
    onnx_model = onnxmltools.convert_keras(model, target_opset=9)

    # save converted models
    with open('{}.onnx'.format(args.onnxmodel_name), 'wb') as f:
        f.write(onnx_model.SerializeToString())
    with open('{}_ctc.onnx'.format(args.onnxmodel_name), 'wb') as f:
        f.write(onnx_ctcmodel.SerializeToString())
