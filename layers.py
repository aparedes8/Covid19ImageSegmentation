
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x):
        y = x
        for layer in self.layers:
            y = layer.train(y)
        return y
        
    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

#############

class layer:
    layer_id = 0
    weight_id = 0

    def __init__(self):
        assert(False)
        
    def train(self, x):        
        assert(False)

    def get_params(self):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, shape):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.k, _, self.f1, self.f2 = shape

        self.f = tf.Variable(init_filters(size=[self.k,self.k,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32, name='f_%d' % (self.layer_id))
        self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32, name='b_%d' % (self.layer_id))
        self.g = tf.Variable(np.ones(shape=(self.f2)), dtype=tf.float32, name='g_%d' % (self.layer_id))

    def train(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,1,1,1], 'SAME')
        mu, var = tf.nn.moments(conv, axes=[0,1,2])
        bn = tf.nn.batch_normalization(conv, mu, var, self.b, self.g, 1e-5)
        relu = tf.nn.relu(bn)
        return relu

    def get_params(self):
        return [self.f, self.b, self.g]

#############
        
class final_conv_block(layer):
    def __init__(self, shape):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.k, _, self.f1, self.f2 = shape

        self.f = tf.Variable(init_filters(size=[self.k,self.k,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32, name='f_%d' % (self.layer_id))
        self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32, name='b_%d' % (self.layer_id))
        self.g = tf.Variable(np.ones(shape=(self.f2)), dtype=tf.float32, name='g_%d' % (self.layer_id))

    def train(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,1,1,1], 'SAME')
        mu, var = tf.nn.moments(conv, axes=[0,1,2])
        bn = tf.nn.batch_normalization(conv, mu, var, self.b, self.g, 1e-5)
        sig = tf.nn.sigmoid(bn)
        return sig

    def get_params(self):
        return [self.f, self.b, self.g]

#############


class max_pool(layer):
    def __init__(self, k):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.k = k

    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.k, strides=self.k, padding="SAME")
        return pool

    def get_params(self):
        return []

#############

class up_pool(layer):
    def __init__(self, k):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.k = k

    def train(self, x):
        shape = tf.shape(x)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        A = x
        A = tf.reshape(A, (b, h * w, 1, c))
        A = tf.tile(A, [1, 1, self.k * self.k, 1])
        A = tf.reshape(A, (b, h, w, self.k, self.k, c))
        A = tf.reshape(A, (b, h, w * self.k, self.k, c))
        A = tf.transpose(A, (0, 1, 3, 2, 4))
        A = tf.reshape(A, (b, h * self.k, w * self.k, c))
        return A
        
    def get_params(self):
        return []

        
#############






        
        
