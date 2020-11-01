# vkCaptchaBreaker
VKontakte captcha bypass with pseudoCRNN model running as chrome extension, Python, JS, 2020

<p align="center">
  <img src="assets/vkcaptcha.gif" alt="title" width="100%"/>    
</p>

HTML Live Demo: [🔥](https://csb-inxnl.netlify.app/)

Codesandbox link for html with onnx js: [__vk-captcha-breaker__](https://codesandbox.io/s/vk-captcha-breaker-inxnl?file=/index.html)

Chrome extension can be downloaded here: [__VKCaptchaBreaker.crx__](https://github.com/Defasium/VKCaptchaBreaker/raw/main/VKcaptcha_breaker.crx)

____

## Table of Contents
  * [Description](#description)
  * [Results](#results)
  * [Sources](#sources)
____
## Description

Keras implementation of Convolutional Recurrent Neural Network with ONNX.js support.
To achieve better accuracy, [`CRNN`](python/crnn.py) utilizes CTC loss and Knowledge distillation with Label smoothing.
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

____
## Results

After [`training`](python/FilthyCaptchaLearning.ipynb) on 1.5 Million filthy labeled captcha for 300k steps, [__student model__](https://github.com/Defasium/models/blob/main/captcha_model.onnx) achieved ~90% accuracy (while teacher accuracy was 99%)
<p align="center">
  <img src="assets/smallKD30_280k.png" alt="training on filthy data"/>
</p>

Comparision:

|Model|Dataset Size|Steps|Knowledge Distillation, %|Accuracy, %|Size, MB|Images/sec|
|:--------------|:--:|:--:|:--:|---:|---:|---:|
|BIG CRNN|5k|70k|-|80|66|~284|
|BIG CRNN|16k|70k|-|93|66|~284|
|BIG CRNN|40k|70k|-|__98.9__|66|~284|
|small pseudoCRNN|40k|70k|-|64|4|~2155|
|small pseudoCRNN|850k*|70k|70|82|4|~2155|
|small pseudoCRNN|850k*|70k|50|83|4|~2155|
|small pseudoCRNN|1.5M*|140k|50|86.7|4|~2155|
|small pseudoCRNN|1.5M*|280k|30|91.2|__4__|__~2155__|

\* - means that dataset was made with filthy labels from teacher

Optimised pseudoCRNN student architecture after [`convertion to ONNX format`](python/onnx_crnn_conversion.py):

<p align="center">
  <img src="assets/captcha_model.onnx.png" alt="architecture"/>
</p>

____

## Sources

This project was inspired by number-plate recognition work and uses its modified version of CRNN implementation, [@RIA.com](https://github.com/ria-com/nomeroff-net).

ONNX.js reference example, created by [@elliotwaite](https://github.com/elliotwaite/pytorch-to-javascript).

For small pseudoCRNN CNN encoder this system uses modified version of ShuffleNetV2, implemented by [@opconty](https://github.com/opconty/keras-shufflenetV2).

Tensorflow Dataset class with caching implementation is the modified version of [@krasserm](https://github.com/krasserm/super-resolution).
