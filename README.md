# vkCaptchaBreaker
Use this VKontakte captcha breaker with 91% accuracy right in browser with chrome extension, Python, JS, 2020

<p align="center">
  <img src="assets/vkcaptcha.gif" alt="title" width="100%"/>    
</p>

### HTML Live Demo from above picture: [__🔥DEMO🔥__](https://csb-inxnl.netlify.app/)

### Code Sandbox for HTML with ONNX.js: [__vk-captcha-breaker__](https://codesandbox.io/s/vk-captcha-breaker-inxnl?file=/index.html)

Images in Demo or Sandbox are broken? [__Solution__](https://github.com/Defasium/vkCaptchaBreaker/issues/3)

### Extension can be downloaded here: 

[__GitHub Releases__](https://github.com/Defasium/vkCaptchaBreaker/releases)

[__Firefox Browser Addons__](https://addons.mozilla.org/firefox/addon/vkcaptchabreaker/)

### Recognition speed comparison on different devices with different CPU specs (in milliseconds):
<p align="center">
  <img src="assets/speed.png" alt="Recognition speed comparison on different devices"/>
</p>

____

## Table of Contents
  * [Installation](#installation)
    * [Windows/Linux/MacOS](#windowslinuxmacos)
    * [Android](#android)
  * [Description](#description)
  * [DataMining](#datamining)  
  * [ONNXTroubles](#onnxtroubles)
  * [Results](#results)
  * [Sources](#sources)
____
## Installation

# <span style="color:red">Update v1.5: recognition in background, Firefox addon </span>
<span style="color:red">Update v1.4: added support for other captcha scenarios</span>

<span style="color:red">Update v1.3: now it works for mobile domain only (m.vk.com)</span>

Google Chrome Extension and Firefox Addon for VKontakte which utilizes trained AI models to predict captcha from image and automatically enters it.

~~Currently only captcha from messages are supported (no likes or comments for you, my bad)~~

You can download latest version from: [__VKCaptchaBreaker.crx__](https://github.com/Defasium/vkCaptchaBreaker/releases)


### Windows/Linux/MacOS

To install the extension, simply follow this tutorial:

For Chrome: [__www.cnet.com/how-to/how-to-install-chrome-extensions-manually__](https://www.cnet.com/how-to/how-to-install-chrome-extensions-manually/)

For Firefox: [__www.muo.com/tag/how-to-install-firefox-add-ons-manually-even-from-github__](https://www.muo.com/tag/how-to-install-firefox-add-ons-manually-even-from-github/)

Finally, activate it:

<p align="center">
  <img src="assets/chrome_activate.gif" alt="title" width="35%"/>
</p>

### Android

To install the extension, simply follow this tutorial:

[__qwikfone.medium.com/how-to-install-chrome-extension-on-android-3cd89f3a771c__](https://qwikfone.medium.com/how-to-install-chrome-extension-on-android-3cd89f3a771c)

Installation and activation example:

<p align="center">
  <img src="assets/android_activate.gif" alt="title" width="35%"/>
</p>

If something won't work, just reactivate extension.

____
## Description

<p align="center">
  <img src="assets/vkcaptchaformat.png" alt="vkcaptchaformat" width="100%"/>
</p>

Keras implementation of Convolutional Recurrent Neural Network with [`ONNX.js`](https://github.com/microsoft/onnxjs) support.
To achieve better accuracy, [`CRNN`](python/crnn.py) utilizes CTC loss and Knowledge distillation with Label smoothing.
CRNN comes in two versions: the one BIG with recurrent layers (RNNs)
and the other small one without them.

The reason for doing so is that if we want to use trained models on end devices, such as
Mobile devices/Client-side browsers, we want to get a model with minimum number of parameters.

For BIG version Google's Inception like CNN encoder is used with Bidirectional GRUs as RNN decoder.
For small version ShuffleNetV2 encoder is used with 1d convs instead of RNN decoder.

____
## DataMining

To parse capthca you can use any url that vkontakte provided, e.g. https://vk.com/captcha.php?sid=625944628258&s=1
Here `625944628258` is a random seed.
It's important to mention that some seeds will generate old captchas with cyrillic symbols.
Using urls with different urls over 1.5 million images was parsed.

At first 5000 captcha's was manualy labelled (that process take approximately 1 year).
Thus, initial dataset was created, let's call it Dataset5k.
Next, on this dataset BIG CRNN was trained and achieved 80% accuracy on unseen data.
After that using vkontakte account without phone verification more images was automatically labelled in online.
Thus, new dataset was created, and the process continues.

After achieving 99% accuracy with BIG CRNN model, it was used for labelling the rest 1.5 million images in offline.
Datasets with 850k images and 1500k images was constructed this way.
This huge "filthy" datasets (implying that there are mistakes in predictions) were used for training small pseudoCRNN model.

____
## ONNXTroubles

<p align="center">
  <img src="https://raw.githubusercontent.com/onnx/onnx/master/docs/ONNX_logo_main.png" alt="onnxlogo" width="100%"/>
</p>

Unfortunately, in the case of using models on Client-side, e.g. with JavaScript, existing frameworks (Open Neural Network Exchange (ONNX.js)) does not support RNNs.
The only one which supports them is Tensorflow.js, but it is slower.
It's important to note that RNNs also are quite slow due to it's nature.

To mitigate this problem this project introduces a small pseudoCRNN architecture without RNN blocks. Instead of bidirectional Gated Recurrent Units (GRU) as in the BIG version, it uses one-dimensional convolutions. 
Other troubles:
 * Can't use dynamic reshaping, e.g Reshape(None, -1, 32) will crash
 * Can't use 1d Convolutions, instead the following trick was used:
 
 (None, 32, 128) --Reshape--> (None, 32, 1, 128) --Conv2d--> (None, 32, 1, filters) --Reshape--> (None, 32, filters)
 * Can't use Droupout, it's important to reconstruct model without this layer
 * Can't use Casting operation in ONNX.js, e.g. INT32 -> FLOAT32 conversion
 * Can't use Ragged Tensors in ONNX.js, e.g. for CTC algorithm
 * Can't use tf.where, because Nonzero is not implemented in ONNX.js
 
You can check the list of supported operators in ONNX.js [`HERE`](https://github.com/microsoft/onnxjs/blob/master/docs/operators.md)

ONNX is a great thing, but its JavaScript Framework needs more attention from microsoft developers.

As a result, in the next project Tensorflow.js should be used instead of ONNX.js

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
|small pseudoCRNN|850k*|70k|-|82|4|~2155|
|small pseudoCRNN|850k*|70k|50|83|4|~2155|
|small pseudoCRNN|1.5M*|140k|50|86.7|4|~2155|
|small pseudoCRNN|1.5M*|280k|30|91.2|__4**__|__~2155__|

\* - means that dataset was made with filthy labels from teacher

\** - after conversion the final size of the model is 1 MB

Thus, you can achieve up to 10 times faster inference time with small version.

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

For model's graph visualization was used the `Netron` application, by [@lutzroeder](https://github.com/lutzroeder/netron)

