What is this?
=============

This is the source code which is used to generate the stream http://www.twitch.tv/317070
The program can recreate images from the neural networks vgg-16 and vgg-19, both found
[here](http://www.vlfeat.org/matconvnet/pretrained/)
What's even better, it can do this interactively and in real time.

What do I need?
===============

For clarity, this code is for linux only. It is best run on a beefy computer:
At least a hexacore CPU
At least a graphics card with 4GB of memory (e.g. the GTX 680, 980 and the Tesla K40 have been tested)
At least 12 GB of RAM (not tested), 32GB is recommended and tested.

You will need the following libraries installed:

1. Cudnn
2. Pylearn2
3. Theano
4. Lasagne

Warning: setting these up is unfortunately not `sudo apt-get' trivial.

You will also need to download the vgg networks from their website [here http://www.vlfeat.org/matconvnet/pretrained/](http://www.vlfeat.org/matconvnet/pretrained/)
Put the resulting .mat files in the data folder, and run the script 
~~~
python mat2npy.py
~~~
to convert to a data-structure lasagne can use.


What do I do?
=============

You can run the default configuration for streams with
~~~
python train.py
~~~
and for images
~~~
python train_image.py
~~~

You can also run custom configurations using:
~~~
python train.py custom_configuration
~~~


How do I set this up?
=====================
Go to models/default.py to edit the default configuration. You will definitely need to fix the Twitch parameters,
these are used for logging into twitch and setting up the stream.


... and now?
============
This source is provided as is. It will be hard to set up (because of its dependencies) by artists or others who are not familiar with these machine learning libraries. Sorry for that.
I think the biggest benefit will be in the reading of the code and using the ideas behind it.

