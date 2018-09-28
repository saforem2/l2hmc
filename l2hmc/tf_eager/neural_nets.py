"""
Neural nets utility for L2HMC compatible with TensorFlow's eager execution.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.
"""
import tensorflow as tf


class GenericNet(tf.keras.Model):
    """Generic neural net with different initialization scale based on input.

    Args:
        x_dim: dimensionality of observed data
        factor: factor of variance scaling initializer
        n_hidden: number of hidden units
    """

    def __init__(self, x_dim, factor, n_hidden=50):
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
