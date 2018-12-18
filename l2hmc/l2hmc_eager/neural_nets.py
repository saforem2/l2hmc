"""
Neural nets utility for L2HMC compatible with TensorFlow's eager execution.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.
"""
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

from utils.tf_logging import variable_summaries

##############################################################################
# Removed from ConvNet architecture
# ----------------------------------------------------------------------------
#  self.batch_norm_x1 = (
#      tf.keras.layers.BatchNormalization(axis=0, name='batch_x1')
#  )
#  self.batch_norm_v1 = (
#      tf.keras.layers.BatchNormalization(axis=0, name='batch_v1')
#  )
#
#  self.dropout_x1 = tf.keras.layers.Dropout(0.25, name='dropout_x1')
#  self.dropout_v1 = tf.keras.layers.Dropout(0.25, name='dropout_v1')
##############################################################################

class ConvNet(tf.keras.Model):
    """Convolutional neural network with different initializaiton scale based
    on input."""
    def __init__(self, 
                 input_shape, 
                 links_shape,
                 num_links,
                 factor, 
                 spatial_size,
                 num_hidden=200, 
                 num_filters=None, 
                 filter_size1=None,
                 filter_size2=None,
                 name='ConvNet',
                 scope='Net'):
        """Initialization method."""

        super(ConvNet, self).__init__(name=name)

        if num_filters is None:
            num_filters = int(2 * spatial_size)

        if filter_size1 is None:
            filter_size1 = (3, 3)

        if filter_size2 is None:
            filter_size2 = (2, 2)

        #  self.x_dim = np.cumprod(input_shape[1:])[-1]
        self.x_dim = num_links

        self._input_shape = input_shape
        self.links_shape = links_shape

        with tf.variable_scope(scope):
            with tf.variable_scope('conv_x1'):
                self.conv_x1 = tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=filter_size1,
                    activation=tf.nn.relu,
                    #  data_format='channels_first',
                    input_shape=self._input_shape,
                    name='conv_x1'
                )

            with tf.variable_scope('conv_v1'):
                self.conv_v1 = tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=filter_size1,
                    activation=tf.nn.relu,
                    #  data_format='channels_first',
                    input_shape=self._input_shape,
                    name='conv_v1'
                )

            with tf.variable_scope('max_pool_x1'):
                self.max_pool_x1 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_x1'
                )

            with tf.variable_scope('max_pool_v1'):
                self.max_pool_v1 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_v1'
                )

            with tf.variable_scope('conv_x2'):
                self.conv_x2 = tf.keras.layers.Conv2D(
                    filters=2 * num_filters,
                    kernel_size=filter_size2,
                    #  data_format='channels_first',
                    activation=tf.nn.relu,
                    name='conv_x2'
                )

            with tf.variable_scope('conv_x2'):
                self.conv_v2 = tf.keras.layers.Conv2D(
                    filters=2 * num_filters,
                    kernel_size=filter_size2,
                    #  data_format='channels_first',
                    activation=tf.nn.relu,
                    name='conv_v2'
                )

            with tf.variable_scope('max_pool_x2'):
                self.max_pool_x2 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_x2'
                )

            with tf.variable_scope('max_pool_v2'):
                self.max_pool_v2 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_v2'
                )

            self.flatten_x = tf.keras.layers.Flatten(name='flat_x')
            self.flatten_v = tf.keras.layers.Flatten(name='flat_v')

            with tf.variable_scope('v_layer'):
                self.v_layer = _custom_dense(
                    num_hidden,
                    1. / 3.,
                    name='v_layer'
                )

            with tf.variable_scope('x_layer'):
                self.x_layer = _custom_dense(
                    num_hidden,
                    factor / 3.,
                    name='x_layer'
                )

            with tf.variable_scope('t_layer'):
                self.t_layer = _custom_dense(
                    num_hidden,
                    1. / 3.,
                    name='t_layer'
                )

            with tf.variable_scope('h_layer'):
                self.h_layer = _custom_dense(
                    num_hidden,
                    name='h_layer'
                )

            with tf.variable_scope('scale_layer'):
                self.scale_layer = _custom_dense(
                    self.x_dim,
                    0.001,
                    name='scale_layer'
                )

            with tf.variable_scope('coeff_scale'):
                self.coeff_scale = tf.contrib.eager.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_scale',
                    trainable=True
                )
            # Translation
            with tf.variable_scope('translation_layer'):
                self.translation_layer = _custom_dense(
                    self.x_dim,
                    0.001,
                    name='translation_layer'
                )

            with tf.variable_scope('transformation_layer'):
                self.transformation_layer = _custom_dense(
                    self.x_dim,
                    0.001,
                    name='transformation_layer'
                )

            with tf.variable_scope('coeff_transformation'):
                self.coeff_transformation = tf.contrib.eager.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_transformation',
                    trainable=True
                )
            #  _ = [variable_summaries(i) for i in self.trainable_variables]

    def call(self, inputs):
        """Architecture looks like:

            input --> CONV, ReLU --> NORM --> AVG_POOL --> DROPOUT --> 
                CONV, ReLU --> NORM --> MAX_POOL
            
        """
        v, x, t = inputs

        #  v = v.reshape((v.shape[0], *self._input_shape[1:]))
        #  x = x.reshape((x.shape[0], *self._input_shape[1:]))
        #  v = tf.reshape(v, shape=(v.shape[0], *self._input_shape[1:]))
        #  x = tf.reshape(x, shape=(x.shape[0], *self._input_shape[1:]))

        position = self.conv_x1(x)
        position = self.max_pool_x1(position)
        position = self.conv_x2(position)
        position = self.max_pool_x2(position)
        position = self.flatten_x(position)

        momentum = self.conv_v1(v)
        momentum = self.max_pool_v1(momentum)
        momentum = self.conv_v2(momentum)
        momentum = self.max_pool_v2(momentum)
        momentum = self.flatten_v(momentum)


        h = self.v_layer(momentum) + self.x_layer(position) + self.t_layer(t)
        h = tf.nn.relu(h)
        h = self.h_layer(h)
        h = tf.nn.relu(h)

        scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)

        translation = self.translation_layer(h)

        transformation = (
            tf.nn.tanh(self.transformation_layer(h)
                       * tf.exp(self.coeff_transformation))
        )

        scale = tf.reshape(scale, shape=(-1, *self.links_shape))
        translation = tf.reshape(translation, shape=(-1, *self.links_shape))
        transformation = tf.reshape(transformation, shape=(-1,
                                                           *self.links_shape))

        return scale, translation, transformation


class GenericNet(tf.keras.Model):
    """Generic neural net with different initialization scale based on input.

    Args:
        x_dim: dimensionality of observed data
        factor: factor of variance scaling initializer
        n_hidden: number of hidden units
    """

    def __init__(self, x_dim, factor, n_hidden=200):
        super(GenericNet, self).__init__()

        self.v_layer = _custom_dense(n_hidden, 1. / 3.)
        self.x_layer = _custom_dense(n_hidden, factor / 3.)
        self.t_layer = _custom_dense(n_hidden, 1. / 3.)
        self.h_layer = _custom_dense(n_hidden)

        # Scale
        self.scale_layer = _custom_dense(x_dim, 0.001)
        self.coeff_scale = tf.contrib.eager.Variable(
            initial_value=tf.zeros([1, x_dim]), name='coeff_scale',
            trainable=True
        )
        # Translation
        self.translation_layer = _custom_dense(x_dim, factor=0.001)
        # Transformation
        self.transformation_layer = _custom_dense(x_dim, factor=0.001)
        self.coeff_transformation = tf.contrib.eager.Variable(
            initial_value=tf.zeros([1, x_dim]),
            name='coeff_transformation',
            trainable=True
        )

    def call(self, inputs):
        v, x, t = inputs

        h = self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        h = tf.nn.relu(h)
        h = self.h_layer(h)
        h = tf.nn.relu(h)
        scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)
        translation = self.translation_layer(h)
        transformation = (tf.nn.tanh(self.transformation_layer(h))
                          * tf.exp(self.coeff_transformation))
        return scale, translation, transformation


class EmptyNet(tf.keras.Model):
    """Empty neural net for traditional HMC without augmented leapfrog step.

    Equivalent to setting the functions Q, S, T = 0 for x and v in the
    augmented leapfrog operator.

    Args:
        x_dim: dimensionality of observed data
    """
    def __init__(self, x_dim):
        super(GenericNet, self).__init__()

        self.output_layer = tf.constant(0, shape=self.samples.shape,
                                        dtype=tf.float32)

        output = tf.constant(0, shape=self.samples.shape, dtype=tf.float32)
        output = self.flatten_tensor(output)
        return output, output, output

def _custom_dense(units, factor=1., name=None):
    """Custom dense layer with specified weight initialization."""

    return tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=factor * 2.,
            mode='FAN_IN',
            uniform=False
        ),
        bias_initializer=tf.constant_initializer(0., dtype=tf.float32),
        name=name

    )
