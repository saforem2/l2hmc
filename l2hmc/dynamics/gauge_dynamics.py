"""
Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf

from network.conv_net import ConvNet
from network.generic_net import GenericNet

from lattice.lattice import GaugeLattice


class GaugeDynamics(tf.keras.Model):
    """Dynamics engine of naive L2HMC sampler."""
    def __init__(self, lattice, potential_fn, **kwargs):
                 #  beta_init=1.,
                 #  num_steps=10,
                 #  eps=0.1,
                 #  conv_net=True,
                 #  hmc=False,
                 #  eps_trainable=True):
        """Initialization.

        Args:
            lattice: Lattice object containing multiple sample lattices.
            potential_fn: Function specifying minus log-likelihood objective to
                minimize.
            Kwargs (expected):
                beta_init: Initial value of inverse coupling strength, beta.
                    Used as starting point for simulated annealing schedule.
                beta_final: Final value of inverse coupling strength, beta.
                    Used as ending point for simulated annealing schedule.
                num_steps: Number of leapfrog steps to use in integrator.
                eps: Initial step size to use in leapfrog integrator.
                conv_net: Flag specifying whether to use ConvNet architecture
                    or GenericNet architecture. Defaults to True. (Defined in
                    `network/conv_net.py`)
                hmc: Flag indicating whether generic HMC (no augmented
                    leapfrog) should be used instead of L2HMC. Defaults to
                    False.
                eps_trainable: Flag indiciating whether the step size (eps)
                    should be trainable. Defaults to True.
                np_seed: Seed to use for numpy.random.
        """
        super(GaugeDynamics, self).__init__(name='GaugeDynamics')
        #  npr.seed(np_seed)

        self.lattice = lattice
        self.potential = potential_fn
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        for key, val in kwargs.items():
            if key != 'eps':  # want to use self.eps as tf.Variable
                setattr(self, key, val)

        #  if not self.hmc:
        #      alpha = tf.get_variable(
        #          'alpha',
        #          initializer=tf.log(tf.constant(self.eps)),
        #          trainable=self.eps_trainable,
        #      )
        #  else:
        #      alpha = tf.log(tf.constant(self.eps, dtype=tf.float32))

        #  self.eps = tf.exp(alpha, name='eps')
        self.eps = tf.Variable(
            initial_value=kwargs.get('eps', 0.1),
            name='eps',
            dtype=tf.float32,
            trainable=self.eps_trainable
        )
        #
        #  if not tf.executing_eagerly():
        #      self.beta = tf.placeholder(tf.float32, shape=(), name='beta')
        #  else:
        #      self.beta = self.beta_init


        # In the case of two-dimensions, samples_tensor has shape:
        #     [batch_size, time_size, space_size, dim]
        #  self.samples = tf.convert_to_tensor(
        #      self.lattice.samples, dtype=tf.float32  # batch of link configs
        #  )

        self._construct_time()
        self._construct_masks()

        # when performing `tf.reduce_sum` we want to sum over all extra axes.
        # For example, when using conv_net, the input data will have the
        # same shape as self.lattice.samples, so we would want to reduce the
        # sum over the first, second and third axes.
        #  self.axes = np.arange(1, len(self.samples.shape))
        self.axes = np.arange(1, len(self.lattice.samples.shape))

        if self.hmc:
            self.position_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]
            self.momentum_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]

        else:
            if self.conv_net:
                self._build_conv_nets()
            else:
                self._build_generic_nets()

    def _build_conv_nets(self):
        """Build ConvNet architecture for position and momentum functions."""
        #  num_hidden = int(2 * self.x_dim)             # num hidden nodes in MLP
        #  num_filters = 2 * self.lattice.space_size    # num filters in Conv2D
        #  filter_size1 = (3, 3)  # filter size in 1st Conv2D layer
        #  filter_size2 = (2, 2)  # filter size in 2nd Conv2D layer
        #  input_shape = (self.batch_size, *self.lattice.links.shape)

        kwargs = {
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
            'links_shape': self.lattice.links.shape,
            'x_dim': self.lattice.num_links, # dimensionality of target space
            'factor': 2.,
            'spatial_size': self.lattice.space_size,
            'num_hidden': 256,
            'num_filters': int(2 * self.lattice.space_size),
            'filter_sizes': [(2, 2), (2, 2)], # for 1st and 2nd conv. layer
            'name_scope': 'position',
            'data_format': self.lattice.data_format,
        }


        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.position_fn = ConvNet(model_name='XNet', **kwargs)

            kwargs['name_scope'] = 'momentum'
            kwargs['factor'] = 1.
            with tf.name_scope("VNet"):
                self.momentum_fn = ConvNet(model_name='VNet', **kwargs)

    def _build_generic_nets(self):
        """Build GenericNet FC-architectures for position and momentum fns. """

        kwargs = {
            'x_dim': self.x_dim,
            'links_shape': self.lattice.links.shape,
            'num_hidden': int(2 * self.x_dim),
            'name_scope': 'position',
            'factor': 2.
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.position_fn = GenericNet(model_name='XNet', **kwargs)

            kwargs['factor'] = 1.
            kwargs['name_scope'] = 'momentum'
            with tf.name_scope("VNet"):
                self.momentum_fn = GenericNet(model_name='VNet', **kwargs)

    # pylint:disable=too-many-locals
    def apply_transition(self, position, beta):
        """Propose a new state and perform the accept/reject step.

        Args:
            position: Batch of (position) samples (batch of links).
            beta (float): Inverse coupling constant.

        Returns:
            position_post: Proposed position before accept/reject step.
            momentum_post: Proposed momentum before accept/reject step.
            accept_prob: Probability of accepting the proposed states.
            position_out: Samples after accept/reject step.
        """
        # Simulate dynamics both forward and backward
        # Use sample masks to compute the actual solutions
        position_f, momentum_f, accept_prob_f = self.transition_kernel(
            position, beta, forward=True
        )

        position_b, momentum_b, accept_prob_b = self.transition_kernel(
            position, beta, forward=False
        )

        # Decide direction uniformly
        forward_mask = tf.cast(
            tf.random_uniform((self.batch_size,)) > 0.5,
            tf.float32
        )

        backward_mask = 1. - forward_mask

        # Obtain proposed states
        position_post = tf.mod(
            (forward_mask[:, None, None, None] * position_f
             + backward_mask[:, None, None, None] * position_b),
            2 * np.pi
        )

        momentum_post = (forward_mask[:, None, None, None] * momentum_f
                         + backward_mask[:, None, None, None] * momentum_b)

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
            accept_mask[:, None, None, None] * position_post
            + reject_mask[:, None, None, None] * position
        )

        return position_post, momentum_post, accept_prob, position_out

    # pylint:disable=missing-docstring,invalid-name,unused-argument
    def transition_kernel(self, position, beta, forward=True):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        # Resample momentum
        momentum = tf.random_normal(tf.shape(position))

        position_post, momentum_post = position, momentum
        sumlogdet = 0.

        #  Apply augmented leapfrog steps
        for i in range(self.num_steps):
            position_post, momentum_post, logdet = lf_fn(
                position_post,
                momentum_post,
                beta,
                i
            )
            sumlogdet += logdet

        accept_prob = self._compute_accept_prob(
            position,
            momentum,
            position_post,
            momentum_post,
            sumlogdet,
            beta
        )

        return position_post, momentum_post, accept_prob

    def _forward_lf(self, position, momentum, beta, i):
        """One forward augmented leapfrog step."""
        t = self._get_time(i)
        mask, mask_inv = self._get_mask(i)
        sumlogdet = 0.

        momentum, logdet = self._update_momentum_forward(position, momentum,
                                                         beta, t)
        sumlogdet += logdet

        position, logdet = self._update_position_forward(position, momentum,
                                                         t, mask, mask_inv)
        sumlogdet += logdet

        position, logdet = self._update_position_forward(position, momentum,
                                                         t, mask_inv, mask)
        sumlogdet += logdet

        momentum, logdet = self._update_momentum_forward(position, momentum,
                                                         beta, t)
        sumlogdet += logdet

        return position, momentum, sumlogdet

    def _backward_lf(self, position, momentum, beta, i):
        """One backward augmented leapfrog step."""
        # pylint: disable=invalid-name

        # Reversed index/sinusoidal time
        t = self._get_time(self.num_steps - i - 1)
        mask, mask_inv = self._get_mask(self.num_steps - i - 1)
        sumlogdet = 0.

        momentum, logdet = self._update_momentum_backward(position, momentum,
                                                          beta, t)
        sumlogdet += logdet

        position, logdet = self._update_position_backward(position, momentum,
                                                          t, mask_inv, mask)
        sumlogdet += logdet

        position, logdet = self._update_position_backward(position, momentum,
                                                          t, mask, mask_inv)
        sumlogdet += logdet

        momentum, logdet = self._update_momentum_backward(position, momentum,
                                                          beta, t)
        sumlogdet += logdet

        return position, momentum, sumlogdet

    # pylint:disable=invalid-name
    def _update_momentum_forward(self, position, momentum, beta, t):
        """Update v in the forward leapfrog step."""
        #  grad = self.grad_potential(position, beta)
        grad = self.grad_potential(position, beta)

        scale, translation, transformed = self.momentum_fn([position, grad, t])

        scale *= 0.5 * self.eps
        transformed *= self.eps
        momentum = (
            momentum * tf.exp(scale, name='vf_scale')
            - 0.5 * self.eps * (tf.exp(transformed, name='vf_transformed')
                                * grad - translation)
        )

        return momentum, tf.reduce_sum(scale, axis=self.axes)

    # pylint:disable=invalid-name,too-many-arguments
    def _update_position_forward(self, position, momentum, t, mask, mask_inv):
        """Update x in the forward leapfrog step."""

        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )

        scale *= self.eps
        transformed *= self.eps

        position = (
            mask * position
             + mask_inv * (position * tf.exp(scale) + self.eps
                           * (tf.exp(transformed) * momentum + translation))
        )

        return position, tf.reduce_sum(mask_inv * scale, axis=self.axes)

    # pylint:disable=invalid-name
    def _update_momentum_backward(self, position, momentum, beta, t):
        """Update v in the backward leapforg step. Invert the forward update."""
        #  grad = self.grad_potential(position, beta)
        grad = self.grad_potential(position, beta)


        scale, translation, transformed = self.momentum_fn([position, grad, t])

        scale *= -0.5 * self.eps
        transformed *= self.eps
        momentum = (
            tf.exp(scale) * (momentum + 0.5 * self.eps
                             * (tf.exp(transformed) * grad - translation))
        )


        return momentum, tf.reduce_sum(scale, axis=self.axes)

    # pylint:disable=invalid-name
    def _update_position_backward(self, position, momentum, t, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""

        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )

        scale *= -self.eps
        transformed *= self.eps

        position = (
            mask * position
             + mask_inv * tf.exp(scale) * (position - self.eps
                                           * (tf.exp(transformed) * momentum
                                              + translation))
        )

        return position, tf.reduce_sum(mask_inv * scale, axis=self.axes)

    def _compute_accept_prob(self, position, momentum, position_post,
                             momentum_post, sumlogdet, beta):
        """Compute the prob of accepting the proposed state given old state."""
        #  beta = self.lattice.beta
        old_hamil = self.hamiltonian(position, momentum, beta)
        new_hamil = self.hamiltonian(position_post, momentum_post, beta)
        prob = tf.exp(tf.minimum((old_hamil - new_hamil + sumlogdet), 0.))

        # Ensure numerical stability as well as correct gradients
        return tf.where(tf.is_finite(prob), prob, tf.zeros_like(prob))

    def _construct_time(self):
        """Convert leapfrog step index into sinusoidal time."""
        self.ts = []
        for i in range(self.num_steps):
            t = tf.constant(
                [
                    np.cos(2 * np.pi * i / self.num_steps),
                    np.sin(2 * np.pi * i / self.num_steps)

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
        for _ in range(self.num_steps):
            #  Need to use npr here because tf would generate different random
            #  values across different `sess.run`
            idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim // 2]
            mask = np.zeros((self.x_dim,))
            mask[idx] = 1.
            mask = tf.constant(mask, dtype=tf.float32)
            #  if conv_net:
            mask = tf.reshape(mask, shape=self.lattice.links.shape)

            self.masks.append(mask[None, :])

    def _get_mask(self, i):
        """Get binary masks for i-th augmented leapfrog step."""
        m = self.masks[i]
        return m, 1. - m

    def potential_energy(self, position, beta):
        """Compute potential energy using `self.potential` and beta."""
        #  return beta * self.potential(position)
        return tf.multiply(beta, self.potential(position))


    def kinetic_energy(self, v):
        """Compute the kinetic energy."""
        return 0.5 * tf.reduce_sum(v**2, axis=self.axes)

    def hamiltonian(self, position, momentum, beta):
        """Compute the overall Hamiltonian."""
        return (self.potential_energy(position, beta)
                + self.kinetic_energy(momentum))

    def grad_potential(self, position, beta, check_numerics=True):
        """Get gradient of potential function at current location."""
        if tf.executing_eagerly():
            tfe = tf.contrib.eager
            grad_fn = tfe.gradients_function(self.potential_energy,
                                             params=["position"])
            grad = grad_fn(position, beta)[0]
            #  grad_fn = tfe.gradients_function(self.potential_energy, params=[0])
            #  (grad,) = grad_fn(position, beta)
            #  grad = tfe.gradients_function(self.potential_energy)(position)[0]
        else:
            grad = tf.gradients(
                self.potential_energy(position, beta), position
            )[0]
        return grad

    def flatten_tensor(self, tensor):
        """Flattens tensor along axes 1:, since axis=0 indexes sample in batch.

        Example: for a tensor of shape [b, x, y, t] -->
        returns a tensor of shape [b, x * y * t]
        """
        batch_size = tensor.shape[0]
        return tf.reshape(tensor, shape=(batch_size, -1))
