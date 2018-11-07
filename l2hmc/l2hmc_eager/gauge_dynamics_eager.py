"""L2HMC compatible with TensorFlow's eager execution.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf) 

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc. 
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf

#import tensorflow.contrib.eager as tfea
tfe = tf.contrib.eager

from l2hmc_eager import neural_nets
from utils.distributions import quadratic_gaussian, GMM

# pylint: disable invalid-name


class GaugeDynamics(tf.keras.Model):
    """Dynamics engine of naive L2HMC sampler."""

    def __init__(self,
                 lattice,
                 minus_loglikelihood_fn,
                 n_steps=25,
                 eps=0.1,
                 np_seed=1,
                 conv_net=True,
                 test_HMC=False):
        """Initialization.

        Args:
            lattice (array-like):
                Array containing a batch of lattices, each of which is a
                `GaugeLattice` object.
            x_dim: 
                Dimensionality of observed data
                minus_loglikelihood_fn: 
            Log-likelihood function of conditional probability
            n_steps:
                Number of leapfrog steps within each transition
            eps:
                Initial value learnable scale of step size
            np_seed:
                Random seed for numpy; used to control sample masks
        """
        super(GaugeDynamics, self).__init__()

        self.lattice = lattice
        self.samples = tf.convert_to_tensor(np.array(self.lattice.samples),
                                            dtype=tf.float32)

        npr.seed(np_seed)
        self.x_dim = self.lattice.num_links

        #  num_samples = len(self.lattice.samples)

        self.potential = minus_loglikelihood_fn
        self.n_steps = n_steps
        #  self.n_steps = tf.contrib.eager.Variable(
        #      initial_value=n_steps,
        #      name='n_steps',
        #      dtype=tf.int32,
        #      trainable=True
        #  )

        self._construct_time()
        self._construct_masks()


        if not test_HMC:
            if conv_net:
                self.conv_net = conv_net

                input_shape = (1, *self.lattice.links.shape)
                s_size = self.lattice.space_size
                n_hidden = int(2 * self.x_dim)  # num of hidden nodes in FC net
                n_filters = 2 * s_size          # num of filters in Conv2D
                filter_size = (2, 2)            # filter size in Conv2D

                self.position_fn = neural_nets.ConvNet(input_shape,
                                                       factor=2.,
                                                       spatial_size=s_size,
                                                       num_filters=n_filters,
                                                       filter_size=filter_size,
                                                       num_hidden=n_hidden)

                self.momentum_fn = neural_nets.ConvNet(input_shape,
                                                       factor=1.,
                                                       spatial_size=s_size,
                                                       num_filters=n_filters,
                                                       filter_size=filter_size,
                                                       num_hidden=n_hidden)
            else:
                self.conv_net = False

                self.position_fn = neural_nets.GenericNet(self.x_dim,
                                                          factor=2.,
                                                          n_hidden=n_hidden)

                self.momentum_fn = neural_nets.GenericNet(self.x_dim,
                                                          factor=1.,
                                                          n_hidden=n_hidden)
        else:
            self.conv_net = True  # don't use ConvNet architecture
            self.position_fn = self._test_fn  # dummy fn, returns zeros
            self.momentum_fn = self._test_fn  # dummy fn, returns zeros


        self.eps = tf.contrib.eager.Variable(
            initial_value=eps,
            name='eps',
            dtype=tf.float32,
            trainable=True
        )

    def _test_fn(self, *args, **kwargs):
        """Dummy test function used for testing generic HMC sampler.

        Returns: Three identical tensors of zeros, equivalent to setting 
        T_x, Q_x, and S_x to zero in the augmented leapfrog integrator.
        """
        output = tf.constant(0, shape=self.samples.shape, dtype=tf.float32)
        output = self.flatten_tensor(output)
        return output, output, output

    def apply_transition(self, position):
        """Propose a new state and perform the accept/reject step."""
        # Simulate dynamics both forward and backward;
        # Use sampled  masks to compute the actual solutions
        position_f, momentum_f, accept_prob_f = self.transition_kernel(
            position, forward=True
        )
        position_b, momentum_b, accept_prob_b = self.transition_kernel(
            position, forward=False
        )

        # Decide direction uniformly
        batch_size = tf.shape(position)[0]
        forward_mask = tf.cast(tf.random_uniform((batch_size,)) > 0.5,
                               tf.float32)
        backward_mask = 1. - forward_mask

        # Obtain proposed states
        position_post = (
            forward_mask[:, None] * position_f
            + backward_mask[:, None] * position_b
        )

        momentum_post = (
            forward_mask[:, None] * momentum_f
            + backward_mask[:, None] * momentum_b
        )

        # Probability of accepting the proposed states
        accept_prob = (forward_mask * accept_prob_f
                       + backward_mask * accept_prob_b)

        # Accept or reject step
        accept_mask = tf.cast(
            accept_prob > tf.random_uniform(tf.shape(accept_prob)), tf.float32
        )
        reject_mask = 1. - accept_mask

        # Samples after accept / reject step
        position_out = (
            accept_mask[:, None] * position_post
            + reject_mask[:, None] * tf.reshape(position,
                                                shape=(position.shape[0], -1))
        )

        return position_post, momentum_post, accept_prob, position_out

    def transition_kernel(self, position, forward=True):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        # Resample momentum
        momentum = tf.random_normal(tf.shape(position))

        position_post, momentum_post = position, momentum
        sumlogdet = 0.
        # Apply augmented leapfrog steps
        for i in range(self.n_steps):
            position_post, momentum_post, logdet = lf_fn(position_post,
                                                         momentum_post,
                                                         i)
            sumlogdet += logdet
        accept_prob = self._compute_accept_prob(position, momentum,
                                                position_post, momentum_post,
                                                sumlogdet)
        position_post = tf.reshape(position_post,
                                   shape=(position_post.shape[0], -1))
        momentum_post = tf.reshape(momentum_post,
                                   shape=(momentum_post.shape[0], -1))

        return position_post, momentum_post, accept_prob

    def _forward_lf(self, position, momentum, i):
        """One forward augmented leapfrog step."""
        t = self._get_time(i)
        mask, mask_inv = self._get_mask(i)
        sumlogdet = 0.

        momentum, logdet = self._update_momentum_forward(position,
                                                         momentum, t)
        sumlogdet += logdet

        position, logdet = self._update_position_forward(position,
                                                         momentum, t, mask,
                                                         mask_inv)
        sumlogdet += logdet

        position, logdet = self._update_position_forward(position,
                                                         momentum, t, mask_inv,
                                                         mask)
        sumlogdet += logdet

        momentum, logdet = self._update_momentum_forward(position,
                                                         momentum, t)
        sumlogdet += logdet

        return position, momentum, sumlogdet

    def _backward_lf(self, position, momentum, i):
        """One backward augmented leapfrog step."""
        # Reversed index/sinusoidal time
        t = self._get_time(self.n_steps - i - 1)
        mask, mask_inv = self._get_mask(self.n_steps - i - 1)
        sumlogdet = 0.

        momentum, logdet = self._update_momentum_backward(position,
                                                          momentum,
                                                          t)
        sumlogdet += logdet

        position, logdet = self._update_position_backward(position, momentum,
                                                          t, mask_inv, mask)
        sumlogdet += logdet

        position, logdet = self._update_position_backward(position, momentum,
                                                          t, mask, mask_inv)
        sumlogdet += logdet

        momentum, logdet = self._update_momentum_backward(position,
                                                          momentum,
                                                          t)
        sumlogdet += logdet

        return position, momentum, sumlogdet

    def _update_momentum_forward(self, position, momentum, t):
        """Update v in the forward leapfrog step."""
        grad = self.grad_potential(position)

        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        if self.conv_net:
            position = self.expand_tensor(position, self.lattice.links.shape)
            grad = self.expand_tensor(grad, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.momentum_fn([position, grad, t])

        # Flatten tensors from shape [b, x, y, t] --> [b, x * y * t]
        if self.conv_net:
            momentum = self.flatten_tensor(momentum)
            grad = self.flatten_tensor(grad)

        scale *= 0.5 * self.eps
        transformed *= self.eps
        momentum = (
            momentum * tf.exp(scale)
            - 0.5 * self.eps * (tf.exp(transformed) * grad - translation)
        )

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_forward(self, position, momentum, t, mask, mask_inv):
        """Update x in the forward leapfrog step."""
        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        if self.conv_net:
            position = self.expand_tensor(position, self.lattice.links.shape)
            momentum = self.expand_tensor(momentum, self.lattice.links.shape)
            mask = self.expand_tensor(mask, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )

        # Flatten tensors from shape [b, x, y, t] --> [b, x * y * t]
        if self.conv_net:
            position = self.flatten_tensor(position)
            momentum = self.flatten_tensor(momentum)
            mask = self.flatten_tensor(mask)

        scale *= self.eps
        transformed *= self.eps

        position = (
            mask * position
            + mask_inv * (position * tf.exp(scale) + self.eps *
                          (tf.exp(transformed) * momentum + translation))
        )

        return (position, tf.reduce_sum(mask_inv * scale, axis=1))

    def _update_momentum_backward(self, position, momentum, t):
        """Update v in the backward leapforg step. Inverting the forward
        update."""
        grad = self.grad_potential(position)

        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        if self.conv_net:
            position = self.expand_tensor(position, self.lattice.links.shape)
            grad = self.expand_tensor(grad, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.momentum_fn([position, grad, t])

        if self.conv_net:
            # flatten momentum, grad from [b, x, y, t] --> [b, x * y * t]
            momentum = self.flatten_tensor(momentum)
            grad = self.flatten_tensor(grad)

        scale *= -0.5 * self.eps
        transformed *= self.eps
        momentum = (
            tf.exp(scale) * (momentum + 0.5 * self.eps * (tf.exp(transformed) *
                                                          grad - translation))
        )

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_backward(self, position, momentum, t, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""
        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        if self.conv_net:
            position = self.expand_tensor(position, self.lattice.links.shape)
            momentum = self.expand_tensor(momentum, self.lattice.links.shape)
            mask = self.expand_tensor(mask, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )
        # Flatten tensors from shape [b, x, y, t] --> [b, x * y * t]
        if self.conv_net:
            position = self.flatten_tensor(position)
            momentum = self.flatten_tensor(momentum)
            mask = self.flatten_tensor(mask)

        scale *= self.eps
        transformed *= self.eps
        position = (
            mask * position + mask_inv * tf.exp(scale) * (position - self.eps *
                                                          (tf.exp(transformed)
                                                           * momentum +
                                                           translation))
        )

        return (position, tf.reduce_sum(mask_inv * scale, axis=1))

    def _compute_accept_prob(self, position, momentum, position_post,
                             momentum_post, sumlogdet):
        """Compute the prob of accepting the proposed state given old state."""
        #  beta = self.lattice.beta
        old_hamil = self.hamiltonian(position, momentum)
        new_hamil = self.hamiltonian(position_post, momentum_post)
        prob = tf.exp(tf.minimum((old_hamil - new_hamil + sumlogdet), 0.))

        # Ensure numerical stability as well as correct gradients
        return tf.where(tf.is_finite(prob), prob, tf.zeros_like(prob))

    def _construct_time(self):
        """Convert leapfrog step index into sinusoidal time."""
        self.ts = []
        for i in range(self.n_steps):
            t = tf.constant(
                [
                    np.cos(2 * np.pi * i / self.n_steps),
                    np.sin(2 * np.pi * i / self.n_steps)

                ],
                dtype=tf.float32
            )
            self.ts.append(t[None, :])

    def _get_time(self, i):
        """Get sinusoidal time for i-th augmented leapfrog step."""
        return self.ts[i]

    def _construct_masks(self):
        """Construct different binary masks for different time steps."""
        self.masks = []
        for _ in range(self.n_steps):
            #  Need to use npr here because tf would generate different random
            #  values across different `sess.run`
            idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim // 2]
            mask = np.zeros((self.x_dim,))
            mask[idx] = 1.
            mask = tf.constant(mask, dtype=tf.float32)
            self.masks.append(mask[None, :])

    def _get_mask(self, i):
        """Get binary masks for i-th augmented leapfrog step."""
        m = self.masks[i]
        return m, 1. - m

    def kinetic(self, v):
        """Compute the kinetic energy."""
        if len(v.shape) > 1:
            # i.e. v has not been flattened into a vector
            # in this case we want to contract over the axes [1:] to calculate
            # a scalar value for the kinetic energy.
            # NOTE: The first axis of v indexes samples in a batch of samples.
            return 0.5 * tf.reduce_sum(v**2, axis=np.arange(1, len(v.shape)))
        else:
            return 0.5 * tf.reduce_sum(v**2, axis=1)

    def hamiltonian(self, position, momentum):
        """Compute the overall Hamiltonian."""
        return self.potential(position) + self.kinetic(momentum)

    def grad_potential(self, position, check_numerics=True):
        """Get gradient of potential function at current location."""
        if tf.executing_eagerly():
            grad = tfe.gradients_function(self.potential)(position)[0]
        else:
            grad = tf.gradients(self.potential(position), position)[0]
        return tf.convert_to_tensor(grad, dtype=tf.float32)

    def flatten_tensor(self, tensor):
        """Flattens tensor along axes 1:, since axis=0 indexes sample in batch.

        Example: for a tensor of shape [b, x, y, t] -->
        returns a tensor of shape [b, x * y * t]
        """
        batch_size = tensor.shape[0]
        return tf.reshape(tensor, shape=(batch_size, -1))

    def expand_tensor(self, tensor, output_shape):
        """Reshapes tensor of shape
        [batch_size, spatial-size * spatial_size * time_size] ---->
            [batch_size, spatial_size, spatial_size, time_size]
        """
        batch_size = tensor.shape[0]
        return tf.reshape(tensor, shape=(batch_size, *output_shape))


# Loss function
def compute_loss(dynamics, x, scale=0.1, eps=1e-4):
    """Compute loss defined in Eq. (8) of paper."""
    z = tf.random_normal(tf.shape(x))  # Auxiliary variable
    _x, _, x_accept_prob, x_out = dynamics.apply_transition(x)
    _z, _, z_accept_prob, _ = dynamics.apply_transition(z)

    if x.shape != _x.shape:
        x = tf.reshape(x, shape=_x.shape)
    if z.shape != _z.shape:
        z = tf.reshape(z, shape=_z.shape)

    # Add eps for numerical stability; following released implementation
    x_loss = (tf.reduce_sum((tf.math.cos(x) - tf.math.cos(_x))**2, axis=1)
              * x_accept_prob + eps)
    z_loss = (tf.reduce_sum((tf.math.cos(z) - tf.math.cos(_z))**2, axis=1)
              * z_accept_prob + eps)
    #  x_loss = (tf.reduce_sum((x_vec - _x_vec)**2, axis=1)
    #            * x_accept_prob + eps)
    #  x_loss = tf.reduce_sum((x - _x)**2, axis=1) * x_accept_prob + eps
    #  z_loss = tf.reduce_sum((z - _z)**2, axis=1) * z_accept_prob + eps

    loss = tf.reduce_mean(
        (1. / x_loss + 1. / z_loss) * scale - (x_loss + z_loss) / scale, axis=0
    )

    return loss, x_out, x_accept_prob


def loss_and_grads(dynamics, x, loss_fn=compute_loss):
    """Obtain loss value and gradients."""
    with tf.GradientTape() as tape:
        loss_val, out, accept_prob = loss_fn(dynamics, x)
    grads = tape.gradient(loss_val, dynamics.trainable_variables)

    return loss_val, grads, out, accept_prob
