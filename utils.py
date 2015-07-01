import theano
import os
import errno
import numpy as np
import cPickle


floatX = theano.config.floatX
cast_floatX = np.float32 if floatX=="float32" else np.float64


def save_pkl(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    with file(path, 'wb') as f: 
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with file(path, 'rb') as f: 
        obj = cPickle.load(f)
    return obj


def resample_list(list_, size):
    orig_size = len(list_)
    ofs = orig_size//size//2
    delta = orig_size/float(size)
    return [ list_[ofs + int(i * delta)] for i in range(size) ]


def resample_arr(arr, size):
    orig_size = arr.shape[0]
    ofs = orig_size//size//2
    delta = orig_size/float(size)
    idxs = [ofs + int(i * delta) for i in range(size)]
    return arr[idxs]


def asarrayX(value):
    return theano._asarray(value, dtype=theano.config.floatX)


def one_hot(vec, m=None):
    if m is None: m = int(np.max(vec)) + 1
    return np.eye(m)[vec]


def make_sure_path_exists(path):
    """Try to create the directory, but if it already exist we ignore the error"""
    try: os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST: raise


def shared_mmp(data=None, file_name="shm", shape=(0,), dtype=floatX):
    """ Shared memory, only works for linux """
    if not data is None: shape = data.shape
    path = "/dev/shm/lio/"
    make_sure_path_exists(path)
    mmp = np.memmap(path+file_name+".mmp", dtype=dtype, mode='w+', shape=shape)
    if not data is None: mmp[:] = data
    return mmp


def open_shared_mmp(filename,  shape=None, dtype=floatX):
    path = "/dev/shm/lio/"
    return np.memmap(path+filename+".mmp", dtype=dtype, mode='r', shape=shape)


def normalize_zmuv(x, axis=0, epsilon=1e-9):
    """ Zero Mean Unit Variance Normalization"""
    mean = x.mean(axis=axis)
    std = np.sqrt(x.var(axis=axis) + epsilon)
    return (x - mean[np.newaxis,:]) / std[np.newaxis,:]

class struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for
      (k, v) in self.__dict__.iteritems()))
    def keys(self):
        return self.__dict__.keys()