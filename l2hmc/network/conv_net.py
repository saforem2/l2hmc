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

#  from .generic_net import _custom_dense

# pylint:disable=too-many-arguments, too-many-instance-attributes
class ConvNet(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""
    def __init__(self,
                 input_shape,
                 links_shape,
                 num_links,
                 factor,
                 spatial_size,
                 num_hidden=200,
                 num_filters=None,
                 filter_size1=(3, 3),
                 filter_size2=(2, 2),
                 model_name='ConvNet',
                 variable_scope='Net',
                 data_format='channels_last'):
        """Initialization method."""
        super(ConvNet, self).__init__(name=model_name)

        if num_filters is None:
            num_filters = int(2 * spatial_size)

        self.x_dim = num_links
        self._input_shape = input_shape
        self.links_shape = links_shape

        with tf.variable_scope(variable_scope):
            with tf.name_scope('conv_x1'):
                self.conv_x1 = tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=filter_size1,
                    activation=tf.nn.relu,
                    input_shape=self._input_shape,
                    name='conv_x1',
                    dtype=tf.float32,
                    data_format=data_format

                )

                self.max_pool_x1 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_x1'
                )

            with tf.name_scope('conv_v1'):
                self.conv_v1 = tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=filter_size1,
                    activation=tf.nn.relu,
                    input_shape=self._input_shape,
                    name='conv_v1',
                    dtype=tf.float32,
                    data_format=data_format
                )


                self.max_pool_v1 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_x1'
                )

            with tf.name_scope('conv_x2'):
                self.conv_x2 = tf.keras.layers.Conv2D(
                    filters=2*num_filters,
                    kernel_size=filter_size2,
                    activation=tf.nn.relu,
                    name='conv_x2',
                    dtype=tf.float32,
                    data_format=data_format
                )

                self.max_pool_x2 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_x1'
                )

            with tf.name_scope('conv_v2'):
                self.conv_v2 = tf.keras.layers.Conv2D(
                    filters=2 * num_filters,
                    kernel_size=filter_size2,
                    activation=tf.nn.relu,
                    name='conv_v2',
                    dtype=tf.float32,
                    data_format=data_format
                )

                self.max_pool_v2 = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name='max_pool_x1'
                )

            with tf.name_scope('flatten'):
                self.flatten_x = tf.keras.layers.Flatten(name='flat_x')
                self.flatten_v = tf.keras.layers.Flatten(name='flat_v')

            with tf.name_scope('dense_x'):
                self.x_layer = _custom_dense(num_hidden,
                                             factor/3.,
                                             name='x_layer')
            with tf.name_scope('dense_v'):
                self.v_layer = _custom_dense(num_hidden,
                                             1./3.,
                                             name='v_layer')

            with tf.name_scope('dense_t'):
                self.t_layer = _custom_dense(num_hidden,
                                             1./3.,
                                             name='t_layer')

            with tf.name_scope('dense_h'):
                self.h_layer = _custom_dense(num_hidden, name='h_layer')

            with tf.name_scope('dense_scale'):
                self.scale_layer = _custom_dense(self.x_dim,
                                                 0.001,
                                                 name='h_layer')

            with tf.name_scope('dense_translation'):
                self.translation_layer = _custom_dense(
                    self.x_dim,
                    0.001,
                    name='translation_layer'
                )

            with tf.name_scope('dense_transformation'):
                self.transformation_layer = _custom_dense(
                    self.x_dim,
                    0.001,
                    name='transformation_layer'
                )

            #  with tf.name_scope('coeff_scale'):
            self.coeff_scale = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_scale',
                trainable=True,
                dtype=tf.float32
            )

            #  with tf.name_scope('coeff_transformation'):
            self.coeff_transformation = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_transformation',
                trainable=True,
                dtype=tf.float32
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
        x = self.max_pool_x2(self.conv_x2(x))
        #  x = self.flatten(x)
        x = self.flatten_x(x)

        v = self.max_pool_v1(self.conv_v1(v))
        v = self.max_pool_v2(self.conv_v2(v))
        #  v = self.flatten(v)
        v = self.flatten_v(v)

        h = tf.nn.relu(self.v_layer(v) + self.x_layer(x) + self.t_layer(t))
        h = tf.nn.relu(self.h_layer(h))
        #  h = tf.nn.relu(h)

        scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)
        scale = tf.reshape(scale, shape=(-1, *self.links_shape), name='scale')

        translation = tf.reshape(self.translation_layer(h),
                                 shape=(-1, *self.links_shape),
                                 name='translation')

        #  translation = tf.reshape(translation,
        #                           shape=(-1, *self.links_shape),
        #                           name='translation')

        transformation = (self.transformation_layer(h)
                          * tf.exp(self.coeff_transformation))
        transformation = tf.reshape(transformation,
                                    shape=(-1, *self.links_shape),
                                    name='transformation')

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
