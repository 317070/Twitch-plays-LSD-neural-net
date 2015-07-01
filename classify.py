################################################################################
#                                                                          INIT
################################################################################

import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
from time import time, strftime, localtime
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

import utils

# warnings.filterwarnings('ignore', '.*topo.*')

if len(sys.argv) < 2:
    print "Usage: %s <config_path>"%os.path.basename(__file__)
    cfg_path = "models/classification.py"
else: cfg_path = sys.argv[1]

cfg_name = cfg_path.split("/")[-1][:-3]
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

all_params = [x,]

givens = {
    model.input.input_var: x-cfg.mean_img
}

print "Compiling"
compute_output = theano.function([], model.out.get_output(deterministic=True), 
                                    givens=givens, on_unused_input='ignore')

################################################################################
#                                                                         TRAIN
################################################################################

output = np.asarray(compute_output())
print output.shape
print np.argmax(output)
print np.argmax(output,axis=1)
print output[0,np.argmax(output)]
print cfg.class_str[np.argmax(output)]