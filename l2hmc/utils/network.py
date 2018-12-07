# Copyright 2017 Google Inc.

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
    return tuple(output)

def get_kernel_initializer(factor):
    """Create kernel initializer for fully-connected layer with `factor`."""
    return tf.contrib.layers.variance_scaling_initializer(
        factor=factor * 2.,
        mode='FAN_IN',
        uniform=False
    )


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


def network(x_dim, scope, factor, num_nodes=50):
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


class ConvNet2D(object):
    """2D Convolutional neural network using tf.contrib.layers API."""
    def __init__(self,
                 input_shape, 
                 factor, 
                 spatial_size,
                 num_hidden=100, 
                 num_filters=None,
                 filter_size=None, 
                 dropout_keep_prob=0.5,
                 is_training=False,
                 scope='ConvNet'):
        """Convolutional neural network (2D)."""

        if num_filters is None:
            num_filters = int(2 * spatial_size)

        if filter_size is None:
            filter_size = (2, 2)

        new_filter_size = (filter_size[0] + 2, filter_size[0] + 2)
        new_num_filters = num_filters * num_filters

        bias_init = tf.constant_initializer(0., dtype=tf.float32)

        x_kern_init = get_kernel_initializer(factor=factor / 3.)
        v_kern_init = get_kernel_initializer(factor=1./3.)
        scale_kern_init = get_kernel_initializer(factor=0.001)
        transf_kern_init = get_kernel_initializer(factor=0.001)
        transl_kern_init = get_kernel_initializer(factor=0.001)

        self.x_dim = np.cumprod(input_shape[1:])[-1]

        self._input_shape = input_shape

        with tf.variable_scope(scope):
            # Convolutional layer 1
            self.conv_x1 = tf.contrib.layers.conv2d(
                input_shape,
                num_outputs=num_filters,
                kernel_size=filter_size,
                scope='conv_x1'
            )

            self.conv_v1 = tf.contrib.layers.conv2d(
                input_shape,
                num_outputs=num_filters,
                kernel_size=filter_size,
                scope='conv_v1'
            )

            # Max pooling layer 1
            self.pool_x1 = tf.contrib.layers.max_pool2d(
                self.conv_x1,
                kernel_size=[2, 2],
                stride=2,
                scope='pool_x1'
            )

            self.pool_v1 = tf.contrib.layers.max_pool2d(
                self.conv_v1,
                kernel_size=[2, 2],
                stride=2,
                scope='pool_v1'
            )

            # Convolutional layer 2
            self.conv_x2 = tf.contrib.layers.conv2d(
                self.pool_x1,
                num_outputs=2*num_filters,
                filter_size=filter_size,
                activation=tf.nn.relu,
                scope='conv_x2'
            )

            self.conv_v2 = tf.contrib.layers.conv2d(
                self.pool_v1,
                num_outputs=2*num_filters,
                filter_size=filter_size,
                activation=tf.nn.relu,
                scope='conv_v2'
            )

            # Max pooling layer 2
            self.pool_x2 = tf.contrib.layers.max_pool2d(
                self.conv_x2,
                kernel_size=[2, 2],
                stride=2,
                scope='pool_x2'
            )

            self.pool_v2 = tf.contrib.layers.max_pool2d(
                self.conv_v2,
                kernel_size=[2, 2],
                stride=2,
                scope='pool_v2'
            )

            # Convolutional layer 3
            self.fc_x3 = tf.contrib.layers.conv2d(
                self.pool_x2,
                num_outputs=new_num_filters,
                kernel_size=new_filter_size,
                padding='VALID',
                scope='fc_x3'
            )

            self.fc_v3 = tf.contrib.layers.conv2d(
                self.pool_v2,
                num_outputs=new_num_filters,
                kernel_size=new_filter_size,
                padding='VALID',
                scope='fc_v3'
            )

            # Dropout layer
            self.dropout_x = tf.contrib.layers.dropout(
                self.fc_x3,
                dropout_keep_prob,
                is_training=is_training,
                scope='dropout_x'
            )

            self.dropout_v = tf.contrib.layers.dropout(
                self.fc_v3,
                dropout_keep_prob,
                is_training=is_training,
                scope='dropout_v'
            )

            # Flatten layer
            self.flat_x = tf.contrib.layers.flatten(self.dropout_x,
                                                    scope='flat_x')

            self.flat_v = tf.contrib.layers.flatten(self.dropout_v,
                                                    scope='flat_v')

            self.x_layer = tf.layers.dense(
                inputs=self.flat_x,
                units=num_hidden,
                use_bias=True,
                kernel_initializer=x_kern_init,
                bias_initializer=bias_init,
                name='x_layer'
            )

            self.v_layer = tf.layers.dense(
                inputs=self.flat_v,
                units=num_hidden,
                use_bias=True,
                kernel_initializer=v_kern_init,
                bias_initializer=bias_init,
                name='v_layer'
            )

            self.t_layer = Linear(2, num_hidden, scope='t_layer', factor=1./3)
            self.h_layer = Linear(num_hidden, num_hidden, scope='h_layer')

            self.scale_layer = tf.layers.dense(
                inputs=self.h_layer,
                units=self.x_dim,
                use_bias=True,
                kernel_initializer=scale_kern_init,
                bias_initializer=bias_init,
                name='scale_layer'
            )

            self.coeff_scale = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_scale',
                trainable=True
            )

            self.translation_layer = tf.layers.dense(
                inputs=self.h_layer,
                units=self.x_dim,
                use_bias=True,
                kernel_initializer=transl_kern_init,
                bias_initializer=bias_init,
                name='translation_layer'
            )

            self.transformation_layer = tf.layers.dense(
                inputs=self.h_layer,
                units=self.x_dim,
                use_bias=True,
                kernel_initializer=transf_kern_init,
                bias_initializer=bias_init,
                name='transformation_layer'
            )

            self.coeff_transformation = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_transformation',
                trainable=True
            )

    def call(self, inputs):
        v, x, t = inputs

        if x.shape != v.shape:
            x = tf.reshape(x, v.shape)

        h = self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        h = tf.nn.relu(h)
        h = self.h_layer(h)
        h = tf.nn.relu(h)
        scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)
        translation = self.translation_layer(h)
        transformation = (tf.nn.tanh(self.transformation_layer(h))
                          * tf.exp(self.coeff_transformation))
        return scale, translation, transformation


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
                #  variable_summaries(self.W)
                with tf.name_scope('biases'):
                    self.b = tf.get_variable(
                        'b', shape=(out_,),
                        initializer=tf.constant_initializer(0.,
                                                            dtype=TF_FLOAT)
                    )
                    #  variable_summaries(self.b)

    def __call__(self, x):
        self.activations = tf.add(tf.matmul(x, self.W), self.b)

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

