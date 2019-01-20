"""
generic_net.py

Generic, fully-connected neural network architecture for running L2HMC on a
gauge lattice configuration of links.

NOTE: Lattices are flattened before being passed as input to the network.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 01/16/2019
"""
import tensorflow as tf

class GenericNet(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""
    def __init__(self, 
                 x_dim, 
                 links_shape, 
                 factor, 
                 num_hidden=200,
                 model_name='GenericNet',
                 variable_scope='Net'):
        """Initialization method."""

        super(GenericNet, self).__init__(name=model_name)

        self.links_shape = links_shape
        self.x_dim = x_dim

        with tf.variable_scope(variable_scope):
            #  self.flatten_x = tf.keras.layers.Flatten(name='flat_x')
            self.flatten = tf.keras.layers.Flatten(name='flat_v')

            self.x_layer = _custom_dense(num_hidden, factor/3., name='x_layer')
            self.v_layer = _custom_dense(num_hidden, 1./3., name='v_layer')
            self.t_layer = _custom_dense(num_hidden, 1./3., name='t_layar')
            self.h_layer = _custom_dense(num_hidden, name='h_layer')

            self.scale_layer = _custom_dense(self.x_dim, 0.001, name='h_layer')

            self.coeff_scale = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_scale',
                trainable=True,
                dtype=tf.float32,
            )

            self.translation_layer = _custom_dense(
                self.x_dim,
                0.001,
                name='translation_layer'
            )

            self.transformation_layer = _custom_dense(
                self.x_dim,
                0.001,
                name='transformation_layer'
            )

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

            x --> FLATTEN_X --> X_LAYER --> X_OUT
            v --> FLATTEN_V --> V_LAYER --> V_OUT
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

        x = self.flatten(x)
        v = self.flatten(v)
        #  x = self.flatten_x(x)
        #  v = self.flatten_v(v)

        h = self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        h = tf.nn.relu(h)
        h = self.h_layer(h)
        h = tf.nn.relu(h)

        scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)
        scale = tf.reshape(scale, shape=(-1, *self.links_shape), name='scale')

        translation = self.translation_layer(h)
        translation = tf.reshape(translation,
                                 shape=(-1, *self.links_shape),
                                 name='translation')

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
