################################################################################
#                                                                          INIT
################################################################################

import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
from time import strftime, localtime
import time
from subprocess import Popen
import sys
import os
import importlib
import warnings
import string
from glob import glob
import cPickle
import platform
import scipy.misc
from zoomingstream import ZoomingStream
from twitch import TwitchOutputStream, TwitchOutputStreamRepeater
from read_the_chat import ChatReader

import utils

# warnings.filterwarnings('ignore', '.*topo.*')

if len(sys.argv) < 2:
    print "Usage: %s <config_path>"%os.path.basename(__file__)
    cfg_path = "default"
else:
    cfg_path = sys.argv[1]

cfg_name = cfg_path.split("/")[-1]
print "Model:", cfg_name
cfg = importlib.import_module("models.%s" % cfg_name)

expid = "%s-%s-%s" % (cfg_name, platform.node(), strftime("%Y%m%d-%H%M%S", localtime()))
print "expid:", expid


################################################################################
#                                                               BUILD & COMPILE
################################################################################
print "Building"

model = cfg.build_model()

pretrained_params = cfg.pretrained_params

nn.layers.set_all_param_values(model.out, pretrained_params)

all_layers = nn.layers.get_all_layers(model.out)

num_params = nn.layers.count_params(model.out)
print "  number of parameters: %d" % num_params
print "  layer output shapes:"
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    print "    %s %s" % (name, layer.get_output_shape(),)


x = nn.utils.shared_empty(dim=len(model.input.get_output_shape()))
x.set_value(cfg.image.astype("float32").reshape((1,)+cfg.image.shape))

interesting_features = theano.shared(np.array(range(cfg.n_classes), dtype='int32'))
interesting_features.set_value(np.array(range(cfg.n_classes), dtype='int32'))


all_params = [x,]


def l_from_network(inp, pool=1):
    input_shape = x.get_value().shape
    inp = inp[:,:,:input_shape[2]//pool*pool,:input_shape[3]//pool*pool]
    inp = inp.reshape((inp.shape[0],
                       inp.shape[1],
                       inp.shape[2]/pool,
                       pool,
                       inp.shape[3]/pool,
                       pool))
    inp = inp.mean(axis=(3,5))

    network_output = model.to_strengthen.get_output(inp-cfg.mean_img)
    output_shape = model.to_strengthen.get_output_shape()
    return (-( network_output[0,interesting_features[0],output_shape[2]/2:,:]).mean() #first feature on lower half
            -( network_output[0,interesting_features[1],:output_shape[2]/2,:]).mean() #second feature on upper half
           + ( network_output[0,:,:,:]).mean() #other classes should be low!
            )



def l_with_meanpool_student(inp, pool=1):

    w = np.load("student_prior_filters.npy").astype("float32")
    w = np.transpose(w, axes=(3,2,0,1))
    input_shape = x.get_value().shape

    #downsample inp
    inp = inp[:,:,:input_shape[2]//pool*pool,:input_shape[3]//pool*pool]
    inp = inp.reshape((inp.shape[0],
                       inp.shape[1],
                       inp.shape[2]/pool,
                       pool,
                       inp.shape[3]/pool,
                       pool))
    inp = inp.mean(axis=(3,5))

    z = T.nnet.conv2d(inp - 128.0, theano.shared(w),  subsample=(1,1),
                                     border_mode="valid")

    mu = theano.shared(np.load("student_prior_mean.npy").astype("float32"))

    v = 0.665248
    l = (z-mu.dimshuffle("x",0,"x","x"))**2
    l = T.log(1. + l / v)

    return l.mean()



def l_with_meanpool_gaussian(inp, pool=1):

    w = np.load("prior_filters.npy").astype("float32")
    w = np.transpose(w, axes=(3,2,0,1))
    input_shape = x.get_value().shape

    #downsample inp
    inp = inp[:,:,:input_shape[2]//pool*pool,:input_shape[3]//pool*pool]
    inp = inp.reshape((inp.shape[0],
                       inp.shape[1],
                       inp.shape[2]/pool,
                       pool,
                       inp.shape[3]/pool,
                       pool))
    inp = inp.mean(axis=(3,5))
    z = T.nnet.conv2d(inp - 128.0, theano.shared(w),  subsample=(1,1),
                                          border_mode="valid")

    mu = theano.shared(np.load("prior_mean.npy").astype("float32"))
    l = T.sqr(z-mu.dimshuffle("x",0,"x","x"))
    l = T.sqr( (z-mu.dimshuffle("x",0,"x","x"))[:, :-1] )
    return l.mean()


pool_sizes = [1,4,8,16,32]
l = np.float32(cfg.prior_strength) * sum([l_with_meanpool_student(x,pool=p) for p in pool_sizes]) / len(pool_sizes)

pool_sizes = [1]
n = sum([l_from_network(x,pool=p) for p in pool_sizes]) / len(pool_sizes)

train_loss = (n + l)

learning_rate = theano.shared(utils.cast_floatX(cfg.learning_rate))

if hasattr(cfg, 'build_updates'):
    updates, resets = cfg.build_updates(train_loss, all_params, learning_rate)
else:
    updates = nn.updates.sgd(    train_loss, all_params,
                                 learning_rate, )
    resets = []

givens = {
    # target_var: T.sqr(y),
    model.input.input_var: x-cfg.mean_img
}

print "Compiling"
idx = T.lscalar('idx')
iter_train = theano.function([idx], [train_loss,l], givens=givens, updates=updates, on_unused_input='ignore')
compute_output = theano.function([idx], model.to_strengthen.get_output(deterministic=True), givens=givens, on_unused_input='ignore')

################################################################################
#                                                                         TRAIN
################################################################################

n_updates = 0

print "image shape:", x.get_value().shape
files = glob("result/*.png")
for f in files: os.remove(f)

def normalize(img, new_min=0, new_max=255):
    """ normalize numpy array """
    old_min = img.min()
    return 1.*(img-old_min)*(new_max-new_min)/(img.max()-old_min)+new_min

e = 0
chat_reader = ChatReader()

with ZoomingStream(zoomspeed=cfg.zoomspeed,
                   width=cfg.width,
                   height=cfg.height,
                   estimated_input_fps=cfg.estimated_input_fps,
                   fps=25) as stream:
    while True:
        if not e % cfg.steps_per_zoom:
            features, string = chat_reader.process_the_chat()
            print "features activated: "
            print features
            newframe = stream.send_frame( np.transpose(x.get_value()[0]/255.0,(1,2,0)), text=string )
            if newframe is None:
                #there is a problem. Commit harakiri
                time.sleep(5)
                chat_reader.send_chat_message("Faq, there is something wrong with the AI. It might be gaining consciousness. I'm rebooting just to make sure it's dead. Stream might be down for a minute or so! Don't forget to refresh, and tell the others. BRB")
                sys.exit(1)

            features = (list(features)*2)[:2]
            interesting_features.set_value(np.array(features, dtype='int32'))
            #interesting_features_one_hot.set_value( np.eye(cfg.n_classes, dtype='float32')[features].T )

            x.set_value(np.transpose(255*newframe, (2,0,1)).astype("float32").reshape((1,)+cfg.image.shape))
            for reset in resets:
                reset[0].set_value(reset[1])
            #img = np.round(np.clip(img,0.1,254.9))
            #scipy.misc.imsave('result/result%s.png'%(str(e).zfill(4),), img.astype("uint8"))


        loss, l = iter_train(0)
        x_val = x.get_value()
        x.set_value(np.clip(x_val, 0.0, 255.0))

        print e, loss, ((x.get_value()-cfg.image)**2).mean(), l
        e+=1


