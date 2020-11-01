#!/usr/bin/python3
'''Keras implementation of Convolutional Recurrent Neural Network with ONNX.js support.
In this module the CRNN architecture was implemented.
To achieve better accuracy, CRNN utilizes CTC loss and Knowledge distillation with Label smoothing.
CRNN comes in two versions: the one BIG with recurrent layers (RNNs)
and the other small one without them.

The reason for doing so is that if we want to use trained models on end devices, such as
Mobile devices/Client-side browsers, we want to get a model with minimum number of parameters.
Unfortunately, in the case of using models on Client-side, e.g. with JavaScript, existing
frameworks (Open Neural Network Exchange (ONNX.js)) does not support RNNs.
The only one which supports them is Tensorflow.js, but it is slower.
It's important to note that RNNs also are quite slow due to it's nature.

To mitigate this problem this module introduces a small pseudoCRNN architecture without RNN blocks.
Instead of bidirectional Gated Recurrent Units (GRU) as in the BIG version, it uses one-dimensional
convolutions. However, Conv1d blocks haven't been supported yet in ONNX.js,
so instead 2d convolutions with one-dimensional kernels was used.
Thus, you can achieve up to 10 times faster inference time with small version.

For BIG version Google's Inception like CNN encoder is used with Bidirectional GRUs as RNN decoder.
For small version ShuffleNetV2 encoder is used with 1d convs instead of RNN decoder.

Examples:
    To use this module, import it in your script or jupyter notebook with the following command:
        # from crnn import get_model
    To get BIG version for training on (128, 64, 3) images with 21 unique characters use:
        # model = get_model(training=True, input_shape=(64, 128, 3), num_classes=23)
    To get BIG version for inference on (128, 64, 3) images with 21 unique characters use:
        # model = get_model(training=False, input_shape=(64, 128, 3), num_classes=23)
    To get small version for training on (128, 64, 3) images with 21 unique characters use:
        # model = get_model(training=True, small=True, input_shape=(64, 128, 3), num_classes=23)
    To get small version for inference on (128, 64, 3) images with 21 unique characters use:
        # model = get_model(training=False, small=True, input_shape=(64, 128, 3), num_classes=23)
    You should use option 'teacher=True' with 'alpha' hyperparameter on training,
    if knowledge distillation will be used:
        # model = get_model(training=True, teacher=True, alpha=0.5, ...)
    If you want to prepare your model for ONNX conversion, use onnx=True:
        # model = get_model(training=False, onnx=True, ...)
Todo:
    * Add kullback-leibler divergence loss as an alternative to crossentropy
    * Add more options
    * Add Transformer architecture with self-attention as an alternative to RNN and Conv1d
.. _VKontakte captcha bypass with pseudoCRNN model running as chrome extension:
   https://github.com/Defasium/vkCaptchaBreaker
'''


import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.layers.merge import concatenate, add
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Reshape, \
                                           BatchNormalization, Conv2D, MaxPooling2D, Lambda
from tensorflow.python.keras.losses import categorical_crossentropy
from recognition_microarchitectures import MiniGoogLeNet
from shufflenetv2 import ShuffleNetV2


# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    '''Computes Connectionist Temporal Classification (CTC) loss on tensors.
        Args:
            args (tuple os size 4): tuple with Tensors in the following order:
                * y_pred (tensorflow.Tensor(None, None, n_unique_chars+2)) - predicted text by model
                * labels (tensorflow.Tensor(None, None)) - true text
                * input_length (tensorflow.Tensor(None, 1)) - length of the predicted sequence
                    (depends on model structure)
                * label_length (tensorflow.Tensor(None, 1)) - length of the true sequence
                    (depends labels' length)
        Returns:
            loss (tensorflow.Tensor(1,)): CTC loss.

    '''
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# # Loss and train functions, network architecture
def kd_lambda_func(args):
    '''Computes Knowledge Distillation (KD) loss on tensors with Label Smoothing.
        Args:
            args (tuple os size 2): tuple with Tensors in the following order:
                * y_pred (tensorflow.Tensor(None, None, n_unique_chars+2)))
                    - predicted text by model
                * y_teacher (tensorflow.Tensor((None, None, n_unique_chars+2)))
                    - predicted text by teacher
        Returns:
            loss (tensorflow.Tensor(1,)): KD loss.

    '''
    y_pred, y_teacher = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    y_shape = y_teacher.get_shape().as_list()
    smoothing = tf.random.uniform((1, y_shape[1], 1), 0.05, 0.2)
    y_teacher_smoothed = (1 - smoothing) * y_teacher + smoothing / y_shape[-1]
    y_teacher = y_teacher_smoothed[:, 2:, :]

    return categorical_crossentropy(y_teacher, y_pred)

# # Loss and train functions, network architecture
def get_model(training, input_shape=(64, 128, 3), num_classes=23,
              small=False, teacher=False, alpha=0.7, onnx=False):
    '''Constructs CRNN model, depending on optins provided.
        Args:
            training (bool): if True then constructs a trainable model.
            input_shape (tuple of size 3, default is (64, 128, 3)):
              shape of images -> height, width, depth.
            num_classes (int, default is 23): number of unique characters in text plus 2
              (1 for whitespace and 1 for CTC delimeter symbols).
            small (bool, default is False): if True then constructs pseudoCRNN without RNN
              if False constructs BIG CRNN with Bidirectional GRUs.
            teacher (bool, default is False): if True then use additional input
              and mix CTC with Knowledge Distillation losses.
            alpha (float, default is 0.7): is used when teacher is True. Defines the proportion
              of CTC in final loss. The value of 0.7 means that CTC is 70% and KD is 30%.
            onnx (bool, default is False): constructs model compatible with ONNX.js:
                * Input shape will be flattened to 1d
                * Input tensor will be automatically normalized by 255
                * Argmax operaion will be applied to output tensor
        Returns:
            Model (tensorflow.keras.models.Model): Keras Model with different number
              of inputs and outputs, depending on options:
                * if training is True and teacher is False -> 4 inputs, 1 output
                * if training and teacher is True -> 5 inputs, 1 output
                * if training is False -> 1 input, 1 output

    '''
    # Make Networkw
    if onnx:
        inputs = Input(name='the_input', shape=(input_shape[0] * input_shape[1] * input_shape[2],),
                       dtype='float32')
        # (None, 64*128*4)
        inputs2 = Reshape(target_shape=input_shape, name='reshape_input')(inputs)
        # (None, 64, 128, 4)
        inputs2 = Lambda(lambda x: x[:, :, :, :3] / 255., name='norm_input')(inputs2)
        # (None, 64, 128, 3)
    else:
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')
        # (None, 64, 128, 3)
        inputs2 = inputs

    if small:
        inputs2, inner = ShuffleNetV2(include_top=False, input_tensor=inputs2,
                                      num_shuffle_units=[2, 2, 2], pooling=False,
                                      bottleneck_ratio=0.35, activation='elu', scale_factor=2.0)
        # (None, 2, 4, 512)
    else:
        # Convolution layer (VGG)
        inner = MiniGoogLeNet.inceptionv1_module(inputs2, 32, 64, trainable=training)
        # (None, 128, 64, 96)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 96)

        inner = MiniGoogLeNet.inceptionv1_module(inner, 64, 128, trainable=training)
        # (None, 64, 32, 192)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 192)

        inner = MiniGoogLeNet.inceptionv1_module(inner, 128, 256, trainable=training)
        # (None, 32, 16, 384)
        inner = MiniGoogLeNet.inceptionv1_module(inner, 128, 256, trainable=training)
        # (None, 32, 16, 384)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
        # (None, 32, 8, 384)

        inner = MiniGoogLeNet.inceptionv1_module(inner, 256, 512, trainable=training)
        # (None, 32, 8, 768)
        inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)
        # (None, 32, 4, 784)

        inner = Conv2D(512, (2, 2), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                       name='con7')(inner)
        # (None, 16, 8, 512)
        inner = BatchNormalization(trainable=training)(inner)
        inner = Activation('relu')(inner)

    # CNN to RNN
    if small:
        inner = Reshape(target_shape=((32, 1, 128)), name='reshape')(inner)
        # (None, 32, 1, 128)
        if training:
            inner = Dropout(0.05)(inner)
        inner = Dense(64, activation='elu', kernel_initializer='he_normal',
                      name='dense1')(inner)
        # (None, 32, 1, 64)
    else:
        inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)
        # (None, 32, 2048)
        if training:
            inner = Dropout(0.05)(inner)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal',
                      name='dense1')(inner)
        # (None, 32, 64)

    # RNN layer
    if small:

        gru1_merged = concatenate([Conv2D(64, (3, 1), activation='elu',
                                          padding='same', name='gru_1')(inner),
                                   Conv2D(64, (5, 1), activation='elu',
                                          padding='same', dilation_rate=2, name='gru_1b')(inner)])
        # (None, 32, 1, 128)
        gru1_merged = BatchNormalization(trainable=training,
                                         epsilon=1e-5,
                                         momentum=0.1)(gru1_merged)
        if training:
            gru1_merged = Dropout(0.05)(gru1_merged)

        gru2_merged = concatenate([Conv2D(64, (3, 1), activation='elu',
                                          padding='same', name='gru_2')(gru1_merged),
                                   Conv2D(64, (5, 1), activation='elu',
                                          padding='same', dilation_rate=2,
                                          name='gru_2b')(gru1_merged)])
        # (None, 32, 1, 128)
        gru2_merged = BatchNormalization(trainable=training,
                                         epsilon=1e-5,
                                         momentum=0.1)(gru2_merged)
        gru2_merged = Reshape(target_shape=((32, 128)), name='reshape2')(gru2_merged)
        # (None, 32, 128)
    else:
        gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal'
                    , name='gru1')(inner)
        # (None, 32, 256)
        gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                     name='gru1_b')(inner)
        reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)

        gru1_merged = add([gru_1, reversed_gru_1b])
        # (None, 32, 512)
        gru1_merged = BatchNormalization(trainable=training)(gru1_merged)

        gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal',
                    name='gru2')(gru1_merged)
        gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                     name='gru2_b')(gru1_merged)
        reversed_gru_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_2b)

        gru2_merged = concatenate([gru_2, reversed_gru_2b])
        # (None, 32, 512)
        gru2_merged = BatchNormalization(trainable=training)(gru2_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(gru2_merged)
    # (None, 32, 23)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[5], dtype='float32')
    # (None, 5)
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels,
                                                                       input_length, label_length])
    # (None, 1)
    if teacher:
        y_teacher = Input(name='teacher_preds', shape=[32, num_classes], dtype='float32')
        kd_loss = Lambda(kd_lambda_func, output_shape=(1,), name='kd')([y_pred, y_teacher])
        # (None, 1)
        loss_out = Lambda(lambda l: alpha * l[0] + (1 - alpha) * l[1], output_shape=(1,),
                          name='loss')([loss_out, kd_loss])
    if training:
        inputs = [inputs, labels, input_length, label_length]
        if teacher:
            inputs.append(y_teacher)
            return Model(inputs=inputs, outputs=loss_out)
        return Model(inputs=inputs, outputs=loss_out)

    # else inference
    if onnx:
        y_pred = Lambda(lambda x: x[:, 2:], name='slice')(y_pred)
        y_pred = Lambda(lambda x: tf.argmax(x, dimension=-1), name='argmax')(y_pred)
    model = Model(inputs=[inputs], outputs=y_pred)
    for i, _ in enumerate(model.layers):
        model.layers[i].trainable = False
    return model
