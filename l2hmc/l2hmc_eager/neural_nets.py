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

class ConvNet(tf.keras.Model):
    """Convolutional neural network with different initializaiton scale based
    on input."""
    def __init__(self, input_shape, factor, spatial_size,
                 num_hidden=200, num_filters=None, filter_size=None):
        super(ConvNet, self).__init__(name='conv_net')
        if num_filters is None:
            num_filters = int(2 * spatial_size)
            #  num_filters = 2 * input_shape[1]
        if filter_size is None:
            filter_size = (2, 2)

        chan_dim = -1  # `channel` dim, i.e. temporal dim of lattice

        self.x_dim = np.cumprod(input_shape[1:])[-1]
        #  self.input_shape = input_shape
        self._input_shape = input_shape

        self.conv_x1 = tf.keras.layers.Conv2D(filters=num_filters,
                                              kernel_size=filter_size,
                                              activation=tf.nn.relu,
                                              input_shape=self._input_shape)
        self.conv_v1 = tf.keras.layers.Conv2D(filters=num_filters,
                                              kernel_size=filter_size,
                                              activation=tf.nn.relu,
                                              input_shape=self._input_shape)
        self.batch_norm_x1 = tf.keras.layers.BatchNormalization(axis=chan_dim)
        self.batch_norm_v1 = tf.keras.layers.BatchNormalization(axis=chan_dim)
        #  self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=2)
        self.dropout_x1 = tf.keras.layers.Dropout(0.25)
        self.dropout_v1 = tf.keras.layers.Dropout(0.25)
        self.max_pool_x1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                        strides=2)
        self.max_pool_v1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                        strides=2)
        self.conv_x2 = tf.keras.layers.Conv2D(filters=2 * num_filters,
                                              kernel_size=filter_size,
                                              activation=tf.nn.relu)

        self.conv_v2 = tf.keras.layers.Conv2D(filters=2 * num_filters,
                                              kernel_size=filter_size,
                                              activation=tf.nn.relu)
        self.flatten_x = tf.keras.layers.Flatten()
        self.flatten_v = tf.keras.layers.Flatten()

        self.v_layer = _custom_dense(num_hidden, 1. / 3.)
        self.x_layer = _custom_dense(num_hidden, factor / 3.)
        self.t_layer = _custom_dense(num_hidden, 1. / 3.)

        self.h_layer = _custom_dense(num_hidden)

        self.scale_layer = _custom_dense(self.x_dim, 0.001)

        self.coeff_scale = tf.contrib.eager.Variable(
            initial_value=tf.zeros([1, self.x_dim]), name='coeff_scale',
            trainable=True
        )
        # Translation
        self.translation_layer = _custom_dense(self.x_dim, 0.001)
        self.transformation_layer = _custom_dense(self.x_dim, factor=0.001)
        self.coeff_transformation = tf.contrib.eager.Variable(
            initial_value=tf.zeros([1, self.x_dim]),
            name='coeff_transformation',
            trainable=True
        )

    def call(self, inputs):
        """Architecture looks like:

            input --> CONV, ReLU --> NORM --> AVG_POOL --> DROPOUT --> 
                CONV, ReLU --> NORM --> MAX_POOL
            
        """
        v, x, t = inputs
        #  h = self.v_model(v) + self.x_model(x) + self.t_layer(t)
        #  v_shape = v.shape
        #  x_shape = x.shape
        #  #  t_shape = t.shape
        #  if len(v.shape) == 2:
        #      v = tf.reshape(v, shape=(v.shape[0], *self.input_shape))
        #  if len(x.shape) == 2:
        #      x = tf.reshape(x, shape=(x.shape[0], *self.input_shape))
        #
        v_conv = self.max_pool_v1(self.conv_v1(v))
        v_conv = self.flatten_v((self.conv_v2(v_conv)))
        x_conv = self.max_pool_x1(self.conv_x1(x))
        x_conv = self.flatten_x((self.conv_x2(x_conv)))
        #  v_conv = self.flatten_v((self.conv_v2(self.max_pool_v1(self.conv_v1(v)))))
        #  v_conv = self.flatten((self.conv_2(self.max_pool(self.conv_1(v)))))
        #  x_conv = self.flatten((self.conv_2(self.max_pool(self.conv_1(x)))))
        h = self.v_layer(v_conv) + self.x_layer(x_conv) + self.t_layer(t)
        h = tf.nn.relu(h)
        h = self.h_layer(h)
        scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)
        translation = self.translation_layer(h)
        transformation = (tf.nn.tanh(self.transformation_layer(h)
                                     * tf.exp(self.coeff_transformation)))

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


def _custom_dense(units, factor=1.):
    """Custom dense layer with specified weight initialization."""

    return tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=factor * 2.,
            mode='FAN_IN',
            uniform=False
        ),
        bias_initializer=tf.constant_initializer(0., dtype=tf.float32)
    )
