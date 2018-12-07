"""
Dynamics object
"""

##############################################################################
#  TODO:
# ============================================================================
#    * Implement correct updates for LatticeDynamics._forward_step and
#        _backward_step, using Gauge transformations to update link variables.
# ----------------------------------------------------------------------------
#    * Finish implementing:
#        - LatticeDynamics._backward_step
#        - LatticeDynamics.forward
#        - LatticeDynamics.backward
# ----------------------------------------------------------------------------
#    * Implement `lattice_propose` in lattice_sampler.py using the
#        LatticeDynamics code.
##############################################################################

import tensorflow as tf
import numpy as np
from functools import reduce
#  from lattice.gauge_lattice import GaugeLattice
from l2hmc_eager import neural_nets

tfe = tf.contrib.eager

TF_FLOAT = tf.float32
NP_FLOAT = np.float32

NUM_AUX_FUNCS = 3

def safe_exp(x, name=None):
    return tf.exp(x)
    return tf.check_numerics(tf.exp(x), message=f'{name} is NaN')

def cast_f32(tensor):
    return tf.cast(tensor, dtype=tf.float32)


class GaugeDynamics(object):
    """Dynamics engine of naive L2HMC sampler."""
    def __init__(self,
                 lattice,
                 minus_loglikelihood_fn,
                 n_steps=10,
                 eps=0.1,
                 hmc=False,
                 conv_net=True,
                 eps_trainable=True,
                 np_seed=1):
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

        self.lattice = lattice
        self.potential = minus_loglikelihood_fn
        self.n_steps = n_steps
        self.conv_net = conv_net

        self.x_dim = self.lattice.num_links

        self.samples = tf.convert_to_tensor(
            np.array(self.lattice.samples),
            dtype=tf.float32
        )

        self.batch_size = self.lattice.samples.shape[0]
        #  self._feature_dims = len(self.lattice.links.shape)
        #  self._transition_mask_shape = tuple([self.batch_size] +
        #                                      [1] * self._feature_dims)

        np.random.seed(np_seed)

        #  self._construct_time()
        self._construct_masks()

        self.eps = tf.Variable(
            initial_value=eps,
            trainable=eps_trainable,
            name=None,
            dtype=tf.float32
        )

        if not hmc:
            if conv_net:
                self._build_conv_nets()
            else:
                self._build_generic_nets()
        else:
            self.position_fn = self._test_fn
            self.momentum_fn = self._test_fn

    def _build_conv_nets(self):
        """Build ConvNet architecture for position and momentum functions."""

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
                                               num_hidden=n_hidden,
                                               name='XConvNet',
                                               scope='XNet')

        self.momentum_fn = neural_nets.ConvNet(input_shape,
                                               factor=1.,
                                               spatial_size=s_size,
                                               num_filters=n_filters,
                                               filter_size=filter_size,
                                               num_hidden=n_hidden,
                                               name='VConvNet',
                                               scope='VNet')

    def _build_generic_nets(self):
        """Build GenericNet FC-architectures for position and momentum fns."""

        n_hidden = int(2 * self.x_dim)  # num of hidden nodes in FC net

        self.position_fn = neural_nets.GenericNet(self.x_dim,
                                                  factor=2.,
                                                  n_hidden=n_hidden)

        self.momentum_fn = neural_nets.GenericNet(self.x_dim,
                                                  factor=1.,
                                                  n_hidden=n_hidden)

    def _test_fn(self, *args, **kwargs):
        """Dummy test function used for testing generic HMC sampler.
        
        Returns:
            Three identical tensors of zeros, equivalent to setting the
            auxiliary functions T, Q, and S to zero in the augmented leapfrog
            integrator.
        """
        output = tf.constant(0, shape=self.samples.shape, dtype=tf.float32)
        output = self.flatten_tensor(output)
        return output, output, output

    def apply_transition(self, position):
        """Propose a new state and perform the accept/reject step."""

        # Simulate dynamics both forward and backward;
        # Use sampled masks to compute the actual solutions
        position_f, momentum_f, accept_prob_f = self.transition_kernel(
            position, forward=True
        )
        position_b, momentum_b, accept_prob_b = self.transition_kernel(
            position, forward=False
        )

        # Decide direction uniformly
        #  batch_size = tf.shape(position)[0]
        forward_mask = tf.cast(tf.random_uniform((self.batch_size,)) > 0.5,
                               tf.float32)

        #  forward_mask = tf.reshape(forward_mask,
        #                            shape=self._transition_mask_shape)

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

        #  accept_mask = tf.reshape(accept_mask,
        #                           shape=self._transition_mask_shape)

        reject_mask = 1. - accept_mask

        # Samples after accept / reject step
        position_out = (
            accept_mask[:, None] * position_post
            + reject_mask[:, None] * position
        )

        return position_post, momentum_post, accept_prob, position_out

    def transition_kernel(self, position, forward=True):
        """Transition kerel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        # Resample momentum
        momentum = tf.random_normal(tf.shape(position), dtype=tf.float32)
        position_post, momentum_post = position, momentum

        # Apply augmented leapfrog steps
        #  batch_size = tf.shape(position)[0]
        sumlogdet = tf.zeros((self.batch_size,))

        t = tf.constant(0., dtype=tf.float32)

        # pylint: disable=unused-argument, missing-docstring
        def body(x, v, t, sumlogdet):
            new_x, new_v, logdet = lf_fn(x, v, t)

            return new_x, new_v, t + 1, sumlogdet + logdet

        def cond(x, v, t, sumlogdet):  
            return tf.less(t, self.n_steps)

        position_post, momentum_post, i, sumlogdet = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[position_post, momentum_post, t, sumlogdet]
        )

        accept_prob = self._compute_accept_prob(position, momentum,
                                                position_post, momentum_post,
                                                sumlogdet)

        #  position_post = self.flatten_tensor(position_post)
        #  momentum_post = self.flatten_tensor(momentum_post)
        #  position_post = tf.reshape(position_post,
        #                             shape=(position_post.shape[0], -1))
        #  momentum_post = tf.reshape(momentum_post,
        #                             shape=(momentum_post.shape[0], -1))
        return position_post, momentum_post, accept_prob

    def _forward_lf(self, position, momentum, i):
        """One forward augmented leapfrog step."""
        #  t = self._get_time(i)
        t = self._format_time(i, tile=tf.shape(position)[0])
        mask, mask_inv = self._get_mask(i)
        sumlogdet = 0.

        momentum, logdet = self._update_momentum_forward(position, momentum, t)

        sumlogdet += logdet

        position, logdet = self._update_position_forward(position, momentum, t,
                                                         mask, mask_inv)
        sumlogdet += logdet

        position, logdet = self._update_position_forward(position, momentum, t,
                                                         mask_inv, mask)
        sumlogdet += logdet

        momentum, logdet = self._update_momentum_forward(position, momentum, t)

        sumlogdet += logdet

        return position, momentum, sumlogdet

    def _backward_lf(self, position, momentum, i):
        """One backward augmented leapfrog step."""
        # Reversed index/sinusoidal time
        #  t = self._get_time(self.n_steps - i - 1)
        t = self._format_time(i, tile=tf.shape(position)[0])
        mask, mask_inv = self._get_mask(self.n_steps - i - 1)
        sumlogdet = 0.

        momentum, logdet = self._update_momentum_backward(position,
                                                          momentum,
                                                          t)
        sumlogdet += logdet

        position, logdet = self._update_position_backward(position,
                                                          momentum,
                                                          t, mask_inv, mask)
        sumlogdet += logdet

        position, logdet = self._update_position_backward(position,
                                                          momentum,
                                                          t, mask, mask_inv)
        sumlogdet += logdet

        momentum, logdet = self._update_momentum_backward(position,
                                                          momentum,
                                                          t)
        sumlogdet += logdet

        return position, momentum, sumlogdet

    def _update_momentum_forward(self, position, momentum, t):
        """Update momentum `v` in the forward leapfrog step."""
        grad = self.grad_potential(position)

        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        #  if self.conv_net:
        #      position = self.expand_tensor(position, self.lattice.links.shape)
        #      grad = self.expand_tensor(grad, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.momentum_fn([position, grad, t])

        # Flatten tensors from shape [b, x, y, t] --> [b, x * y * t]
        #  if self.conv_net:
        #      momentum = self.flatten_tensor(momentum)
        #      grad = self.flatten_tensor(grad)

        scale *= 0.5 * self.eps
        transformed *= self.eps

        momentum = (
            momentum * tf.exp(scale)
            - 0.5 * self.eps * (tf.exp(transformed) * grad - translation)
        )

        #  axes = np.arange(1, len(scale.shape))
        #  return momentum, tf.reduce_sum(scale, axis=axes)
        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_forward(self, position, momentum, t, mask, mask_inv):
        """Update position `x` in the forward leapfrog step."""
        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        #  if self.conv_net:
        #      position = self.expand_tensor(position, self.lattice.links.shape)
        #      momentum = self.expand_tensor(momentum, self.lattice.links.shape)
        #      mask = self.expand_tensor(mask, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )

        # Flatten tensors from shape [b, x, y, t] --> [b, x * y * t]
        #  if self.conv_net:
        #      position = self.flatten_tensor(position)
        #      momentum = self.flatten_tensor(momentum)
        #      mask = self.flatten_tensor(mask)

        scale *= self.eps
        transformed *= self.eps

        position = (
            mask * position
            + mask_inv * (position * tf.exp(scale) + self.eps
                          * (tf.exp(transformed) * momentum + translation))
        )
        #  axes = np.arange(1, len(scale.shape))
        #  return position, tf.reduce_sum(mask_inv * scale, axis=axes)

        return position, tf.reduce_sum(mask_inv * scale, axis=1)

    def _update_momentum_backward(self, position, momentum, t):
        """Update momentum `v` in the backward leapfro step.

        Inverting the forward update.
        """
        grad = self.grad_potential(position)

        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        #  if self.conv_net:
        #      position = self.expand_tensor(position, self.lattice.links.shape)
        #      grad = self.expand_tensor(grad, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.momentum_fn([position, grad, t])

        #  if self.conv_net:
        #      # flatten momentum, grad from [b, x, y, t] --> [b, x * y * t]
        #      momentum = self.flatten_tensor(momentum)
        #      grad = self.flatten_tensor(grad)

        scale *= -0.5 * self.eps
        transformed *= self.eps

        momentum = (
            tf.exp(scale) * (momentum + 0.5 * self.eps * (tf.exp(transformed) *
                                                          grad - translation))
        )

        #  axes = np.arange(1, len(scale.shape))
        #  return momentum, tf.reduce_sum(scale, axis=axes)
        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_backward(self, position, momentum, t, mask, mask_inv):
        """Update position `x` in the backward lf step. 
        
        Inverting the forward update.
        """
        # Reshape tensors to satisfy input shape for convolutional layer
        # want to reshape from [b, x * y * t] --> [b, x, y, t]
        #  if self.conv_net:
        #      position = self.expand_tensor(position, self.lattice.links.shape)
        #      momentum = self.expand_tensor(momentum, self.lattice.links.shape)
        #      mask = self.expand_tensor(mask, self.lattice.links.shape)

        # scale, translation, transformed all have shape [b, x * y * t]
        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )

        # flatten tensors from shape [b, x, y, t] --> [b, x * y * t]
        #  if self.conv_net:
        #      position = self.flatten_tensor(position)
        #      momentum = self.flatten_tensor(momentum)
        #      mask = self.flatten_tensor(mask)

        scale *= self.eps
        transformed *= self.eps
        position = (
            mask * position + mask_inv * tf.exp(scale)
            * (position - self.eps * (tf.exp(transformed)
                                      * momentum + translation))
        )
        #  position = (
        #      mask * position + mask_inv * tf.exp(scale) * (
        #          position - self.eps * (
        #              tf.exp(transformed) * momentum + translation
        #          )
        #      )
        #  )

        return position, tf.reduce_sum(mask_inv * scale, axis=1)
        #  axes = np.arange(1, len(scale.shape))
        #  return position, tf.reduce_sum(mask_inv * scale, axis=axes)

    def _compute_accept_prob(self, position, momentum, 
                             position_post, momentum_post, sumlogdet):
        """Compute the prob of accepting the proposed state given old state."""
        old_hamil = self.hamiltonian(position, momentum)
        new_hamil = self.hamiltonian(position_post, momentum_post)

        prob = tf.exp(tf.minimum((old_hamil - new_hamil + sumlogdet), 0.))

        # Ensure numerical stability as well as correct gradients
        return tf.where(tf.is_finite(prob), prob, tf.zeros_like(prob))

    def _format_time(self, t, tile=1):
        trig_time = tf.squeeze([
            tf.cos(2 * np.pi * t / self.n_steps),
            tf.sin(2 * np.pi * t / self.n_steps),
        ])
        return tf.tile(tf.expand_dims(trig_time, 0), (tile, 1))

    #  def _construct_time(self):
    #      """Convert leapfrog step index into sinusoidal time."""
    #      self.ts = []
    #      for i in range(self.n_steps):
    #          t = tf.constant(
    #              [
    #                  np.cos(2 * np.pi * i / self.n_steps),
    #                  np.sin(2 * np.pi * i / self.n_steps)
    #              ],
    #              dtype=tf.float32
    #          )
    #          self.ts.append(t[None, :])
    #
    #  def _get_time(self, i):
    #      """Get sinusoidal time for i-th augmented leapfrog step."""
    #      return self.ts[i]

    def _construct_masks(self):
        """Construct different binary masks for different time steps."""
        mask_per_step = []
        for _ in range(self.n_steps):
            # Need to use np.random here because tf would generate different
            # random values across different `sess.run`
            idx = np.random.permutation(np.arange(self.x_dim))[:self.x_dim //
                                                               2]
            mask = np.zeros((self.x_dim,))
            mask[idx] = 1.
            #  mask = tf.reshape(mask, shape=self.lattice.links.shape)

            mask_per_step.append(mask[None, :])

            self.mask = tf.constant(np.stack(mask_per_step), dtype=tf.float32)

        #  self.mask = tf.reshape(self.mask, shape=(*self.mask.shape[:-1],
        #                                           *self.lattice.links.shape))

    def _get_mask(self, step):
        """Get mask at time step `step`."""
        m = tf.gather(self.mask, tf.cast(step, dtype=tf.int32))
        return m, 1. - m

    #  def _get_mask(self, i):
    #      """Get binary masks for i-th augmented leapfrog step."""
    #      m = self.masks[i]
    #      return m, 1. - m

    def kinetic(self, v):
        """Compute the kinetic energy."""
        if len(v.shape) > 1:
            # i.e. v has not been flattened into a vector
            # in this case, we want to contract over the axes [1:] to calculate
            # a scalar value for the kinetic energy.
            # NOTE: The first axis of v indexes samples in a batch of samples.
            return 0.5 * tf.reduce_sum(v ** 2, axis=np.arange(1, len(v.shape)))
        else:
            return 0.5 * tf.reduce_sum(v ** 2, axis=1)

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
        """Flattens along axes [1:], since axis=0 indexes samples in batch.

        Example:
            For a tensor of shape [b, x, y, t] -->
            returns a tensor of shape [b, x * y * t].
        """
        batch_size = tensor.shape[0]
        return tf.reshape(tensor, shape=(batch_size, -1))

    def expand_tensor(self, tensor, output_shape):
        """Expands tensor along all but the first axis.

        Example:
            For a tensor of shape [b, x * y * t] -->
            returns a tensor of shape [b, x, y, t].
        """
        batch_size = tensor.shape[0]
        return tf.reshape(tensor, shape=(batch_size, *output_shape))
