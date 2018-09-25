# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of useful layers
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from .tf_logging import variable_summaries
from .unitary_utils import *

TF_FLOAT = tf.float32
NP_FLOAT = np.float32

def flatten(tup):
    output = []
    for i in tup:
        if hasattr(i, "__iter__") or hasattr(i, "__len__"):
            output += flatten(i)
        else:
            output += [i]
        #  output += flatten(i) if hasattr(i, "__iter__") or hasattr(i, "__len__")
        #  else[i]
    return tuple(output)

def complex_network(x_dim, scope, factor, num_nodes=100):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(x_dim, num_nodes, scope='embed_1', factor=1.0/3),
                Linear(x_dim, num_nodes, scope='embed_2', factor=factor*1.0/3),
                Linear(2, num_nodes, scope='embed_3', factor=1.0/3),
                lambda _: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(num_nodes, num_nodes, scope='linear_1'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(num_nodes, x_dim, scope='linear_s', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_s')

                ])
            ])
        ])

###############################################################################
#    Network Architecture:
# -----------------------------------------------------------------------------
#    * Time step t is given as input to MLP, encoded as 
#        tau(t) = (cos(2pi*t/M), sin(2pi*t/M))
#    * For nh hidden units per layer, compute:
#        (1.) h1 = ReLU(W1*x + W2*v + W3*tau(t) + b), h1.shape = (nh,)
#        (2.) h2 = ReLU(W4*h + b4), h2.shape = (nh,)
#        (3.) Sv = lambda_s * tanh(Ws*h2 + bs)
#             Qv = lambda_q * tanh(Wq*h2 + bq)
#             Tv = Wt*h2 + bt
#
#    * Each of Q, S, T are neural networks with 2 hidden layers with nh nodes
###############################################################################

def network(x_dim, scope, factor, num_nodes=100):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(x_dim, num_nodes, scope='embed_1', factor=1.0/3),
                Linear(x_dim, num_nodes, scope='embed_2', factor=factor*1.0/3),
                Linear(2, num_nodes, scope='embed_3', factor=1.0/3),
                lambda _: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(num_nodes, num_nodes, scope='linear_1'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(num_nodes, x_dim, scope='linear_s', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_s')
                ]),
                Linear(num_nodes, x_dim, scope='linear_t', factor=0.001),
                Sequential([
                    Linear(num_nodes, x_dim, scope='linear_f', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_f'),
                ])
            ])
        ])
        return net

def _network(x_dim, scope, factor, num_nodes=100):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(x_dim, num_nodes, scope='embed_1', factor=1.0/3),
                Linear(x_dim, num_nodes, scope='embed_2', factor=factor*1.0/3),
                Linear(2, num_nodes, scope='embed_3', factor=1.0/3),
                lambda _: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(num_nodes, num_nodes, scope='linear_1'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(num_nodes, x_dim, scope='linear_s', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_s')
                ]),
                Linear(num_nodes, x_dim, scope='linear_t', factor=0.001),
                Sequential([
                    Linear(num_nodes, x_dim, scope='linear_f', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_f'),
                ])
            ])
        ])
        return net



class Linear(object):
    def __init__(self, in_, out_, scope='linear', factor=1.0):
        with tf.variable_scope(scope):
            initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=factor * 2.0,
                mode='FAN_IN',
                uniform=False,
                dtype=TF_FLOAT
            )
            with tf.name_scope('weights'):
                shape_tup = (in_, out_)
                _shape = flatten(shape_tup)
                self.W = tf.get_variable(
                    'W', shape=_shape, initializer=initializer,

                )
                variable_summaries(self.W)
                with tf.name_scope('biases'):
                    self.b = tf.get_variable(
                        'b', shape=(out_,),
                        initializer=tf.constant_initializer(0.,
                                                            dtype=TF_FLOAT)
                    )
                    variable_summaries(self.b)

    def __call__(self, x):
        #  self.activations = tf.add(tf.matmul(x, self.W), self.b)
        #self.W = tf.reshape(self.W, shape=tuple([x.shape] + [-1]))
        #  import pdb
        #  pdb.set_trace()
        if x.dtype == tf.complex64:
            self.activations = tf.add(complex_matmul(x, self.W),
                                      tf.cast(self.b, tf.complex64))
            #  self.activations = tf.add(complex_matmul(x, self.W),
            #                            tf.cast(self.b, tf.complex64))
        else:
            self.activations = tf.add(tf.matmul(x, self.W), self.b)

        #  tf.summary.histogram('activations', self.activations)
        #  return tf.add(tf.matmul(x, self.W), self.b)
        return self.activations


class ConcatLinear(object):
    def __init__(self, ins_, out_, factors=None, scope='concat_linear'):
        self.layers = []

        with tf.variable_scope(scope):
            for i, in_ in enumerate(ins_):
                if factors is None:
                    factor = 1.0
                else:
                    factor = factors[i]
                self.layers.append(Linear(in_, out_, scope='linear_%d' % i,
                                          factor=factor))

    def __call__(self, inputs):
        output = 0.
        for i, x in enumerate(inputs):
            output += self.layers[i](x)
        return output


class Parallel(object):
    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        return [layer(x) for layer in self.layers]


class Sequential(object):
    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class ScaleTanh(object):
    def __init__(self, in_, scope='scale_tanh'):
        with tf.variable_scope(scope):
            self.scale = tf.exp(tf.get_variable(
                'scale', shape=(1, in_),
                initializer=tf.constant_initializer(0., dtype=TF_FLOAT)
            ))

    def __call__(self, x):
        return self.scale * tf.nn.tanh(x)


class Zip(object):
    def __init__(self, layers=[]):
        self.layers = layers

    def __call__(self, x):
        assert len(x) == len(self.layers)
        n = len(self.layers)
        return [self.layers[i](x[i]) for i in range(n)]
