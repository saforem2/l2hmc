"""L2HMC compatible with TensorFlow's eager execution.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf

# pylint: disable invalid-name, module level import not at top of file
from . import neural_nets
tfe = tf.contrib.eager


class GaugeDynamicsEager(tf.keras.Model):
    """Dynamics engine of naive L2HMC sampler."""

    def __init__(self,
                 lattice,
                 minus_loglikelihood_fn,
                 num_steps=25,
                 eps=0.1,
                 np_seed=1,
                 conv_net=True,
                 hmc=False,
                 eps_trainable=True):
        """Initialization.

        Args:
            lattice (array-like): Array containing a batch of lattices, each of
                which is a `GaugeLattice` object.
            x_dim: Dimensionality of observed data
            minus_loglikelihood_fn: Log-likelihood function of conditional
                probability
            num_steps: Number of leapfrog steps within each transition
            eps: Initial value learnable scale of step size
            np_seed: Random seed for numpy; used to control sample masks
        """
        super(GaugeDynamicsEager, self).__init__()
        npr.seed(np_seed)

        self.lattice = lattice
        self.batch_size = self.lattice.samples.shape[0]

        self.samples = tf.convert_to_tensor(
            self.lattice.samples, dtype=tf.float32
        )

        self.x_dim = self.lattice.num_links
        self.potential = minus_loglikelihood_fn
        self.num_steps = num_steps
        self.conv_net = conv_net
        self.hmc = hmc

        self._construct_time()
        self._construct_masks()

        # when performing `tf.reduce_sum` we want to sum over all
        # `extra axes`, e.g. when using conv_net, samples has shape
        # (batch_size, time_size, space_size, dimension) so we want to
        # reduce the sum over the first, second, and third axes. 
        self.axes = np.arange(1, len(self.samples.shape))

        if not hmc:
            if conv_net:
                self._build_conv_nets()
            else:
                self._build_generic_nets()
        else:
            #  z = lambda x, *args, **kwargs: tf.zeros_like(x)
            self.position_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]
            self.momentum_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]
            #  self.position_fn = self._test_fn
            #  self.momentum_fn = self._test_fn
            #  self.axes = 1

        if tf.executing_eagerly():
            self.eps = tf.contrib.eager.Variable(
                initial_value=eps,
                name='eps',
                dtype=tf.float32,
                trainable=eps_trainable
            )
        else:
            self.eps = tf.Variable(
                initial_value=eps,
                name='eps',
                dtype=tf.float32,
                trainable=eps_trainable
            )
            tf.add_to_collection('eps', self.eps)

    def _build_conv_nets(self):
        """ Build ConvNet architecture for position and momentum functions. """
        input_shape = (1, *self.lattice.links.shape)
        links_shape = self.lattice.links.shape
        num_links = self.lattice.num_links
        s_size = self.lattice.space_size
        n_hidden = int(2 * self.x_dim)  # num of hidden nodes in FC net
        n_filters = 2 * s_size          # num of filters in Conv2D
        filter_size1 = (3, 3)           # filter size in first Conv2D layer
        filter_size2 = (2, 2)           # filter size in second Conv2D layer

        self.position_fn = neural_nets.ConvNet(input_shape,
                                               self.lattice.links.shape,
                                               self.lattice.num_links,
                                               factor=2.,
                                               spatial_size=s_size,
                                               num_filters=n_filters,
                                               filter_size1=filter_size1,
                                               filter_size2=filter_size2,
                                               num_hidden=int(2*self.x_dim),
                                               model_name='XNet',
                                               variable_scope='position')

        self.momentum_fn = neural_nets.ConvNet(input_shape,
                                               self.lattice.links.shape,
                                               self.lattice.num_links,
                                               factor=1.,
                                               spatial_size=s_size,
                                               num_filters=n_filters,
                                               filter_size1=filter_size1,
                                               filter_size2=filter_size2,
                                               num_hidden=int(2*self.x_dim),
                                               model_name='VNet',
                                               variable_scope='momentum')

    def _build_generic_nets(self):
        """Build GenericNet FC-architectures for position and momentum fns. """
        self.position_fn = neural_nets.GenericNet(self.x_dim,
                                                  self.lattice.links.shape,
                                                  factor=2.,
                                                  n_hidden=int(2*self.x_dim))

        self.momentum_fn = neural_nets.GenericNet(self.x_dim,
                                                  self.lattice.links.shape,
                                                  factor=1.,
                                                  n_hidden=int(2*self.x_dim))

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
        forward_mask = tf.cast(tf.random_uniform((self.batch_size,)) > 0.5,
                               tf.float32)
        backward_mask = 1. - forward_mask

        # Obtain proposed states
        position_post = tf.mod(
            (forward_mask[:, None, None, None] * position_f
            + backward_mask[:, None, None, None] * position_b), 2*np.pi
        )
        momentum_post = (
            forward_mask[:, None, None, None] * momentum_f
            + backward_mask[:, None, None, None] * momentum_b
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
        position_out = tf.mod(
            (accept_mask[:, None, None, None] * position_post
             + reject_mask[:, None, None, None] * position), 2*np.pi
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
        for i in range(self.num_steps):
            position_post, momentum_post, logdet = lf_fn(position_post,
                                                         momentum_post,
                                                         i)
            sumlogdet += logdet

        accept_prob = self._compute_accept_prob(position, momentum,
                                                position_post, momentum_post,
                                                sumlogdet)

        return position_post, momentum_post, accept_prob

    def _forward_lf(self, position, momentum, i):
        """One forward augmented leapfrog step."""
        t = self._get_time(i)
        mask, mask_inv = self._get_mask(i)
        sumlogdet = 0.

        #  print("\t\t\t Running self._update_momentum_forward...")
        momentum, logdet = self._update_momentum_forward(position,
                                                         momentum, t)
        sumlogdet += logdet

        #  print("\t\t\t Running self._update_position_forward...")
        position, logdet = self._update_position_forward(position,
                                                         momentum, t, mask,
                                                         mask_inv)
        sumlogdet += logdet

        #  print("\t\t\t Running self._update_position_forward...")
        position, logdet = self._update_position_forward(position,
                                                         momentum, t, mask_inv,
                                                         mask)
        sumlogdet += logdet

        #  print("\t\t\t Running self._update_momentum_forward...")
        momentum, logdet = self._update_momentum_forward(position,
                                                         momentum, t)
        sumlogdet += logdet

        return position, momentum, sumlogdet

    def _backward_lf(self, position, momentum, i):
        """One backward augmented leapfrog step."""
        # Reversed index/sinusoidal time
        t = self._get_time(self.num_steps - i - 1)
        mask, mask_inv = self._get_mask(self.num_steps - i - 1)
        sumlogdet = 0.

        #  print("\t\t\t Running self._update_momentum_backward...")
        momentum, logdet = self._update_momentum_backward(position,
                                                          momentum,
                                                          t)
        sumlogdet += logdet

        #  print("\t\t\t Running self._update_position_backward...")
        position, logdet = self._update_position_backward(position, momentum,
                                                          t, mask_inv, mask)
        sumlogdet += logdet

        #  print("\t\t\t Running self._update_position_backward...")
        position, logdet = self._update_position_backward(position, momentum,
                                                          t, mask, mask_inv)
        sumlogdet += logdet

        #  print("\t\t\t Running self._update_momentum_backward...")
        momentum, logdet = self._update_momentum_backward(position,
                                                          momentum,
                                                          t)
        sumlogdet += logdet

        return position, momentum, sumlogdet

    def _update_momentum_forward(self, position, momentum, t):
        """Update v in the forward leapfrog step."""
        grad = self.grad_potential(position)

        # scale, translation, transformed all have shape [b, x * y * t]
        #  print("\t\t\t\t Running self.momentum_fn...")
        #  scale, translation, transformed = self.momentum_fn(position, grad, t)
        scale, translation, transformed = self.momentum_fn([position, grad, t])

        scale *= 0.5 * self.eps
        transformed *= self.eps
        momentum = (
            momentum * tf.exp(scale)
            - 0.5 * self.eps * (tf.exp(transformed) * grad - translation)
        )

        return momentum, tf.reduce_sum(scale, axis=self.axes)

    def _update_position_forward(self, position, momentum, t, mask, mask_inv):
        """Update x in the forward leapfrog step."""

        # scale, translation, transformed all have shape [b, x * y * t] ???
        #  print("\t\t\t\t Running self.position_fn...")
        #  scale, translation, transformed = self.position_fn(momentum,
        #                                                     mask*position, t)
        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )

        scale *= self.eps
        transformed *= self.eps

        position = (
            mask * position
            + mask_inv * (position * tf.exp(scale) + self.eps *
                          (tf.exp(transformed) * momentum + translation))
        )

        return position, tf.reduce_sum(mask_inv * scale, axis=self.axes)

    def _update_momentum_backward(self, position, momentum, t):
        """Update v in the backward leapforg step. Inverting the forward
        update."""
        grad = self.grad_potential(position)


        # scale, translation, transformed all have shape [b, x * y * t]
        #  print("\t\t\t\t Running self.momentum_fn`...")
        #  scale, translation, transformed = self.momentum_fn(position, grad, t)
        scale, translation, transformed = self.momentum_fn([position, grad, t])

        scale *= -0.5 * self.eps
        transformed *= self.eps
        momentum = (
            tf.exp(scale) * (momentum + 0.5 * self.eps * (tf.exp(transformed) *
                                                          grad - translation))
        )

        return momentum, tf.reduce_sum(scale, axis=self.axes)

    def _update_position_backward(self, position, momentum, t, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""

        # scale, translation, transformed all have shape [b, x * y * t]
        #  print("\t\t\t\t Running self.position_fn...")
        #  scale, translation, transformed = self.position_fn(momentum,
        #                                                     mask * position, t)
        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )

        scale *= self.eps
        transformed *= self.eps

        position = (
            mask * position + mask_inv * tf.exp(scale)
            * (position - self.eps * (tf.exp(transformed)
                                      * momentum + translation))
        )

        return position, tf.reduce_sum(mask_inv * scale, axis=self.axes)

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

    def kinetic(self, v):
        """Compute the kinetic energy."""
        return 0.5 * tf.reduce_sum(v**2, axis=self.axes)

    def hamiltonian(self, position, momentum):
        """Compute the overall Hamiltonian."""
        return self.potential(position) + self.kinetic(momentum)

    def grad_potential(self, position, check_numerics=True):
        """Get gradient of potential function at current location."""
        if tf.executing_eagerly():
            grad = tfe.gradients_function(self.potential)(position)[0]
        else:
            grad = tf.gradients(self.potential(position), position)[0]
        #  return tf.convert_to_tensor(grad, dtype=tf.float32)
        return grad

    def flatten_tensor(self, tensor):
        """Flattens tensor along axes 1:, since axis=0 indexes sample in batch.

        Example: for a tensor of shape [b, x, y, t] -->
        returns a tensor of shape [b, x * y * t]
        """
        batch_size = tensor.shape[0]
        return tf.reshape(tensor, shape=(batch_size, -1))


def get_new_sample(dynamics, x, transition_fn=None, defun=False):
    """Get new sample from dynamics by passing x through the network."""
    if transition_fn is None:
        #  if defun:
        #      apply_transition = tfe.defun(dynamics.apply_transition)
        #  else:
        apply_transition = dynamics.apply_transition
    else:
        apply_transition = transition_fn

    _x, _, x_accept_prob, x_out = apply_transition(x)

    _x = tf.mod(_x, 2*np.pi)
    x_out = tf.mod(x_out, 2*np.pi)
    return _x, x_accept_prob, x_out


# pylint: disable=invalid-name
def compute_loss(dynamics, x, _x, x_accept_prob, params, defun=False):
    """Compute loss defined in Eq. (8) of paper."""

    scale = params.get('loss_scale', 0.1)
    eps = params.get('loss_eps', 1e-4)
    metric = params.get('metric', 'l2')

    axes = dynamics.axes

    #  z = tf.random_normal(tf.shape(x))  # Auxiliary variable

    #  _x, x_accept_prob, x_out = get_new_sample(dynamics, x, defun)
    #  _z, z_accept_prob, z_out = get_new_sample(dynamics, z, defun)

    # Add eps for numerical stability; following released implementation
    if metric == 'l2':  # l2 Euclidean squared norm
        x_loss = tf.reduce_sum((x - _x)**2, axis=axes) * x_accept_prob + eps
        #  z_loss = tf.reduce_sum((z - _z)**2, axis=axes) * z_accept_prob + eps

    if metric == 'l1':
        x_loss = tf.reduce_sum(tf.abs(x - _x), axis=axes) * x_accept_prob + eps
        #  z_loss = tf.reduce_sum(tf.abs(z - _z), axis=axes) * z_accept_prob + eps

    else:  # `cos` metric (NOTE: experimental!)
        x_loss = (tf.reduce_sum((tf.math.cos(x)
                                 - tf.math.cos(_x))**2, axis=axes)
                  * x_accept_prob + eps)
        #  z_loss = (tf.reduce_sum((tf.math.cos(z)
        #                           - tf.math.cos(_z))**2, axis=axes)
        #            * z_accept_prob + eps)

    loss = tf.reduce_mean(scale / x_loss - x_loss / scale, axis=0)
    #  loss = tf.reduce_mean(
    #      (1. / x_loss + 1. / z_loss) * scale - (x_loss + z_loss) / scale, axis=0
    #  )

    return loss


def loss_and_grads(dynamics, x, params, loss_fn=compute_loss, 
                   hmc=False, transition_fn=None, defun=False):
    """Obtain loss value and gradients.

    Args:
        dynamics: Main dynamics engine responsible for implementing L2HMC alg.
        samples (tf.Tensor): Batch of training samples.
        optimizer: Tensorflow optimizer (e.g. tf.train.AdamOptimizer)
        loss_fn (function): Function that computes loss from network output.
        params (dict): Dictionary of parameters. 
            We are interested in:
                * params['loss_eps']: Small constant for numerical stability.
                * params['metric']: Metric used in calculating loss. 
                * params['loss_scale']: Scaling factor (lambda) used in
                    calculating the loss.

    Returns:
        loss (float): `loss` value output from network.
        gradients: Resulting gradient values from this training step.
        out: Output from Metropolis Hastings accept/reject step (new samples).
        accept_prob (float): Probability that proposed states were accepted.
    """
    if defun:
        _get_new_sample = tfe.defun(get_new_sample)
    else:
        _get_new_sample = get_new_sample

    with tf.GradientTape() as tape:
        _x, x_accept_prob, x_out = _get_new_sample(dynamics, x,
                                                   transition_fn, defun)
        #  _x, x_accept_prob, x_out = dynamics(x)
        loss_val = loss_fn(dynamics, x, _x, x_accept_prob, params, defun)
        #  loss_val, x_out, accept_prob = loss_fn(dynamics, x, params, defun)

    grads = tape.gradient(loss_val, dynamics.trainable_variables)

    return loss_val, x_out, x_accept_prob, grads

