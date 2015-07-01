from functools import partial
import theano
import theano.tensor as T
import numpy as np
import lasagne as nn
import scipy.misc
from glob import glob
from lasagne.layers import dnn

import utils

""" The twitter login keys """
#""" #toggle this comment to use test or live version
#test-version
TWITCH_USERNAME = ""
TWITCH_OAUTH = ""
TWITCH_STREAM_KEY = ""
"""
#live version
TWITCH_USERNAME = ""
TWITCH_OAUTH = ""
TWITCH_STREAM_KEY = ""
#"""

"""The number of streams created simultaneously (1 usually)"""
batch_size = 1

"""Learning rate"""
learning_rate = 5.0
"""Momentum (not used?)"""
momentum = theano.shared(np.float32(0.9))
"""Number of gradient steps per zoom step"""
steps_per_zoom = 10
"""exponential power of the network (not used?)"""
network_power = 1
"""Strength of the image prior (very important!)"""
prior_strength = 10
"""Zoom speed in percentage per second (very important!)"""
zoomspeed = 1.05
"""Width and height of the stream, NOT of the image optimized"""
width = 1024
height = 576 #multiple of 2!
"""Estimated number of optimized frames generated, important in beginning of stream only"""
estimated_input_fps=15./steps_per_zoom
"""Number of classes. Change this when using other networks."""
n_classes = 1000

"""The image used to start with, also determines the size of the image optimized!"""
image = scipy.misc.imread(glob("image7.*")[0])

"""The network used"""
pretrained_params = np.load("data/vgg16.npy")




print image.dtype
mean_img = np.transpose(np.load("data/mean.npy").astype("float32"), axes=(2,0,1)).mean() #.mean() for any size
# image -= mean_img
image = np.transpose(image, axes=(2,0,1))

conv3 = partial(dnn.Conv2DDNNLayer,
    strides=(1, 1),
    border_mode="same", 
    filter_size=(3,3),
    nonlinearity=nn.nonlinearities.rectify)

dense = partial(nn.layers.DenseLayer,
    nonlinearity=nn.nonlinearities.rectify)

max_pool = partial(dnn.MaxPool2DDNNLayer,
    ds=(2,2), 
    strides=(2,2))


def build_model(batch_size=batch_size):
    l_in = nn.layers.InputLayer(shape=(batch_size,)+image.shape)
    l = l_in

    l = conv3(l, num_filters=64)
    l = conv3(l, num_filters=64)

    l = max_pool(l)

    l = conv3(l, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)

    l = max_pool(l)

    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)

    l = max_pool(l)

    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)

    l = max_pool(l)

    l = dnn.Conv2DDNNLayer(l,
                num_filters=4096,
                strides=(1, 1),
                border_mode="valid",
                filter_size=(7,7))
    l = dnn.Conv2DDNNLayer(l,
                num_filters=4096,
                strides=(1, 1),
                border_mode="same",
                filter_size=(1,1))

    l = dnn.Conv2DDNNLayer(l,
                num_filters=n_classes,
                strides=(1,1),
                border_mode="same",
                filter_size=(1,1),
                nonlinearity=None)

    l_to_strengthen = l
    l_out = l

    return utils.struct(
        input=l_in,
        out=l_out,
        to_strengthen=l_to_strengthen)



"""The update step. Here using Adam's method."""
def build_updates(loss, all_params, learning_rate,  beta1=0.9, beta2=0.999,
                   epsilon=1e-8):
    all_grads = theano.grad(loss, all_params)
    updates = []
    resets = []
    t = theano.shared(1) # timestep, for bias correction
    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)) # 1st moment
        vparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)) # 2nd moment

        m = beta1 * grad_i + (1 - beta1) * mparam_i # new value for 1st moment estimate
        v = beta2 * T.sqr(grad_i) + (1 - beta2) * vparam_i # new value for 2nd moment estimate

        m_unbiased = m / (1 - (1 - beta1) ** t.astype(theano.config.floatX))
        v_unbiased = v / (1 - (1 - beta2) ** t.astype(theano.config.floatX))
        w = param_i - learning_rate * m_unbiased / (T.sqrt(v_unbiased) + epsilon) # new parameter values

        updates.append((mparam_i, m))
        updates.append((vparam_i, v))
        updates.append((param_i, w))
        resets.append([mparam_i, np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)])
        resets.append([vparam_i, np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)])
    resets.append([t, 1])
    updates.append((t, t + 1))

    return updates, resets
