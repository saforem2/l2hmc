"""
conv_net.py

Convolutional neural network architecture for running L2HMC on a gauge lattice
configuration of links.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 01/16/2019
"""
import tensorflow as tf
import numpy as np

#  from .generic_net import _custom_dense

# pylint: disable=invalid-name
def create_periodic_padding(samples, filter_size):
    """Create periodic padding for each sample in samples, using filter_size."""
    original_size = np.shape(samples)
    N = original_size[1]  # number of links in lattice
    #  N = np.shape(samples)[1] # number of links in lattice
    padding = filter_size - 1

    samples = tf.reshape(samples, shape=(samples.shape[0], -1))

    x = []
    for sample in samples:
        padded = np.zeros((N + 2 * padding), N + 2 * padding, 2)
        # lower left corner
        padded[:padding, :padding, :] = sample[N-padding:, N-padding:, :]
        # lower middle
        padded[padding:N+padding, :padding, :] = sample[:, N-padding:, :]
        # loewr right corner
        padded[N+padding:, :padding, :] = sample[:padding, N-padding:, :]
        # left side
        padded[:padding, padding: N+padding, :] = sample[N-padding:, :, :]
        # center
        padded[:padding:N+padding, padding:N+padding, :] = sample[:, :, :]
        # right side
        padded[N+padding:, padding:N+padding:, :] = sample[:padding, :, :]
        # top middle
        padded[:padding:N+padding, N+padding:, :] = sample[:, :padding, :]
        # top right corner
        padded[N+padding:, N+padding:, :] = sample[:padding, :padding, :]

        x.append(padded)

    return np.array(x, dtype=np.float32).reshape(*original_size)


# pylint:disable=too-many-arguments, too-many-instance-attributes
class ConvNet(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""
    def __init__(self, model_name, **kwargs):
        """Initialization method."""

        super(ConvNet, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)


        #  with tf.variable_scope(self.variable_scope):
        with tf.name_scope(self.name_scope):

            #  with tf.name_scope('batch_norm'):
            #      self.batch_norm = tf.keras.layers.BatchNormalization(
            #          axis=self.channel_dim
            #      )

            self.coeff_scale = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_scale',
                trainable=True,
                dtype=tf.float32
            )

            self.coeff_transformation = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_transformation',
                trainable=True,
                dtype=tf.float32
            )


            self.conv_x1 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.filter_sizes[0],
                activation=tf.nn.relu,
                input_shape=self._input_shape,
                name='conv_x1',
                dtype=tf.float32,
                data_format=self.data_format

            )

            self.max_pool_x1 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.conv_v1 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.filter_sizes[0],
                activation=tf.nn.relu,
                input_shape=self._input_shape,
                name='conv_v1',
                dtype=tf.float32,
                data_format=self.data_format
            )

            self.max_pool_v1 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.conv_x2 = tf.keras.layers.Conv2D(
                filters=2*self.num_filters,
                kernel_size=self.filter_sizes[1],
                activation=tf.nn.relu,
                name='conv_x2',
                dtype=tf.float32,
                data_format=self.data_format
            )

            self.max_pool_x2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.conv_v2 = tf.keras.layers.Conv2D(
                filters=2 * self.num_filters,
                kernel_size=self.filter_sizes[1],
                activation=tf.nn.relu,
                name='conv_v2',
                dtype=tf.float32,
                data_format=self.data_format
            )

            self.max_pool_v2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.flatten = tf.keras.layers.Flatten(name='flatten')

            self.x_layer = _custom_dense(self.num_hidden, self.factor/3.,
                                         name='x_layer')

            self.v_layer = _custom_dense(self.num_hidden, 1./3.,
                                         name='v_layer')

            self.t_layer = _custom_dense(self.num_hidden, 1./3.,
                                         name='t_layer')

            self.h_layer = _custom_dense(self.num_hidden, name='h_layer')

            #  self.hidden_layer1 = tf.keras.layers.Dense(
            #      2 * self.x_dim,
            #      activation=tf.nn.relu
            #  )

            self.scale_layer = _custom_dense(self.x_dim, 0.001,
                                             name='scale_layer')

            self.translation_layer = _custom_dense(self.x_dim, 0.001,
                                                   name='translation_layer')

            self.transformation_layer = _custom_dense(
                self.x_dim,
                0.001,
                name='transformation_layer'
            )


    # pylint: disable=invalid-name, arguments-differ
    def call(self, inputs):
        """call method.

        NOTE: Architecture looks like 
        
        - inputs: x, v, t

            x --> CONV_X1, MAX_POOL_X1, --> CONV_X1, MAX_POOL_X2 -->
                   FLATTEN_X --> X_LAYER --> X_OUT

            v --> CONV_V1, MAX_POOL_V1, --> CONV_V1, MAX_POOL_V2 -->
                   FLATTEN_V --> V_LAYER --> V_OUT

            t --> T_LAYER --> T_OUT

            X_OUT + V_OUT + T_OUT --> H_LAYER --> H_OUT


        - H_OUT is then fed to three separate layers:

            (1.) H_OUT --> (SCALE_LAYER, TANH) * exp(COEFF_SCALE)

                 output: scale
            
            (2.) H_OUT --> TRANSLATION_LAYER --> TRANSLATION_OUT

                 output: translation

            (3.) H_OUT --> (TRANSFORMATION_LAYER, TANH) 
                            * exp(COEFF_TRANSFORMATION)

                 output: transformation

       Returns:
           scale, translation, transformation
        """
        v, x, t = inputs

        x = self.max_pool_x1(self.conv_x1(x))
        x = tf.nn.local_response_normalization(x)
        x = self.max_pool_x2(self.conv_x2(x))
        x = tf.nn.local_response_normalization(x)
        x = self.flatten(x)

        v = self.max_pool_v1(self.conv_v1(v))
        v = tf.nn.local_response_normalization(v)
        v = self.max_pool_v2(self.conv_v2(v))
        v = tf.nn.local_response_normalization(v)
        v = self.flatten(v)

        h = tf.nn.relu(self.v_layer(v) + self.x_layer(x) + self.t_layer(t))
        h = tf.nn.relu(self.h_layer(h))
        #  h = self.hidden_layer1(h)

        def reshape(t, name):
            return tf.reshape(t, shape=self._input_shape, name=name)

        translation = reshape(self.translation_layer(h), name='translation')

        scale = reshape(
            tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale),
            name='scale'
        )

        transformation = reshape(
            self.transformation_layer(h) * tf.exp(self.coeff_transformation),
            name='transformation'
        )

        #  translation = tf.reshape(self.translation_layer(h),
        #                           shape=self._input_shape,
        #                           name='translation')

        #  scale = (tf.nn.tanh(self.scale_layer(h))
        #           * tf.exp(self.coeff_scale))
        #  scale = tf.reshape(scale, shape=self._input_shape, name='scale')

        #  transformation = (self.transformation_layer(h)
        #                    * tf.exp(self.coeff_transformation))
        #
        #  transformation = tf.reshape(transformation,
        #                              shape=self._input_shape,
        #                              name='transformation')

        return scale, translation, transformation


def _custom_dense(units, factor=1., name=None):
    """Custom dense layer with specified weight intialization."""
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
