"""L2HMC compatible with TensorFlow's eager execution.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf) 

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc. 
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tf_eager import neural_nets

from utils.distributions import quadratic_gaussian, GMM


class LatticeDynamics(tf.keras.Model):
    """Dynamics engine of naive L2HMC sampler."""

    def __init__(self,
                 lattice,
                 minus_loglikelihood_fn,
                 batch_size,
                 n_steps=25,
                 eps=0.1,
                 np_seed=1):
        """Initialization.
        
        Args:
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
        super(LatticeDynamics, self).__init__()

        self.lattice = lattice
        self.batch_size = batch_size

        npr.seed(np_seed)
        self.x_dim = self.lattice.sites.shape
        self.x_flat_dim = self.lattice.num_sites

        self.potential = minus_loglikelihood_fn
        self.n_steps = n_steps

        self._construct_time()
        self._construct_masks()

        self.position_fn = neural_nets.GenericNet(self.x_flat_dim, factor=2.)
        self.momentum_fn = neural_nets.GenericNet(self.x_flat_dim, factor=1.)

        self.eps = tf.contrib.eager.Variable(
            initial_value=eps,
            name='eps',
            dtype=tf.float32,
            trainable=True
        )

    def apply_transition(self, position):
        """Propose a new state and perform the accept/reject step."""

        # Simulate dynamics both forward and backward;
        # Use sampled Bernoulli masks to compute the actual solutions
        position_f, momentum_f, accept_prob_f = self.transition_kernel(
            position, forward=True
        )
        position_b, momentum_b, accept_prob_b = self.transition_kernel(
            position, forward=False
        )

        # Decide direction uniformly
        batch_size = tf.shape(position)[0]
        forward_mask = tf.cast(tf.random_uniform((batch_size,)) > 0.4,
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
        accept_prob = (
            forward_mask * accept_prob_f + backward_mask * accept_prob_b
        )

        # Accept or reject step
        accept_mask = tf.cast(
            accept_prob > tf.random_uniform(tf.shape(accept_prob)), tf.float32
        )
        reject_mask = 1. - accept_mask

        # Samples after accept / reject step
        position_out = (
            accept_mask[:, None] * position_post
            + reject_mask[:, None] * position
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

        return position_post, momentum_post, accept_prob

    def _forward_lf(self, position, momentum, i):
        """One forward augmented leapfrog step."""
        t = self._get_time(i)
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
        scale, transition, transformed = self.momentum_fn([position, grad, t])
        scale *= 0.5 * self.eps
        transformed *= self.eps
        momentum = (
            momentum * tf.exp(scale)
            - 0.5 * self.eps * (tf.exp(transformed) * grad - translation)
        )

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_forward(self, position, momentum, t, mask, mask_inv):
        """Update x in the forward leapfrog step."""

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
        return position, tf.reduce_sum(mask_inv * scale, axis=1)

    def _update_momentum_backward(self, position, momentum, t):
        """Update v in the backward leapforg step. Inverting the forward
        update."""

        grad = self.grad_potential(position)
        scale, translation, transformed = self.momentum_fn([position, grad, t])
        scale *= -0.5 * self.eps
        transformed *= self.eps
        momentum = (
            tf.exp(scale) * (momentum + 0.5 * self.eps * (tf.exp(transformed) *
                                                          grad - translation))
        )

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_backward(self, position, momentum, t, mask, mask_inv):
        """Update x in the backward leapfrog step. Inverting the forward
        update."""

        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )
        scale *= self.eps
        transformed *= self.eps
        position = (
            mask * position + mask_inv * tf.exp(scale) * (position - self.eps *
                                                          (tf.exp(transformed)
                                                           * momentum +
                                                           translation))
        )

        return position, tf.reduce_sum(mask_inv * scale, axis=1)

    def _compute_accept_prob(self, position, momentum, position_post,
                             momentum_post, sumlogdet):
        """Compute the prob of accepting the proposed state given old state."""

        old_hamil = self.hamiltonian(position, momentum)
        new_hamil = self.hamiltonian(position_post, momentum_post)
        prob = tf.exp(tf.minimum(old_hamil - new_hamil + sumlogdet, 0.))

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
            #  idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim // 2]
            #  mask = np.zeros((self.x_dim,))
            #  mask[idx] = 1.
            #  mask = tf.constant(mask, dtype=tf.float32)
            #  self.masks.append(mask[None, :])
            arr = np.arange(self.lattice.num_sites)
            idx = np.random.permutation(arr)[:int(self.lattice.num_sites // 2)]
            mask = np.zeros((self.lattice.num_sites,))
            mask[idx] = 1.
            mask = mask.reshape(self.lattice.sites.shape)
            mask = tf.constant(mask, dtype=tf.float32)
            self.masks.append(mask[None, :])

    def _get_mask(self, i):
        """Get binary masks for i-th augmented leapfrog step."""

        m = self.masks[i]
        return m, 1. - m

    def kinetic(self, v):
        """Compute the kinetic energy."""

        flat_v = v.numpy().reshape(-1, self.x_flat_dim)
        #  axes = np.arange(1, len(v.shape))

        return 0.5 * tf.reduce_sum(flat_v**2, axis=1)

    def hamiltonian(self, position, momentum):
        """Compute the overall Hamiltonian."""

        return (
            self.potential(position, self.batch_size) + self.kinetic(momentum)
        )

    def grad_potential(self, position, check_numerics=True):
        """Get gradient of potential function at current location."""

        if tf.executing_eagerly():
            grad_fn = tfe.gradients_function(self.potential)
            grad = grad_fn(position, self.batch_size)[0]
            #  grad = (
            #      tfe.gradients_function(self.potential,
            #                             params=[0])(position)[0]
            #  )
        else:
            grad = tf.gradients(self.potential(position), position)[0]
        return grad


class GaugeDynamics(tf.keras.Model):
    """Dynamics engine of naive L2HMC sampler."""

    def __init__(self,
                 lattice,
                 minus_loglikelihood_fn,
                 batch_size,
                 n_steps=25,
                 eps=0.1,
                 np_seed=1):
        """Initialization.
        
        Args:
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
        self.batch_size = batch_size

        npr.seed(np_seed)
        self.x_dim = self.lattice.links.shape
        self.x_flat_dim = self.lattice.num_links

        self.potential = minus_loglikelihood_fn
        self.n_steps = n_steps

        self._construct_time()
        self._construct_masks()

        self.position_fn = neural_nets.GenericNet(self.x_flat_dim, factor=2.)
        self.momentum_fn = neural_nets.GenericNet(self.x_flat_dim, factor=1.)

        self.eps = tf.contrib.eager.Variable(
            initial_value=eps,
            name='eps',
            dtype=tf.float32,
            trainable=True
        )

    def apply_transition(self, position):
        """Propose a new state and perform the accept/reject step."""

        # Simulate dynamics both forward and backward;
        # Use sampled Bernoulli masks to compute the actual solutions
        position_f, momentum_f, accept_prob_f = self.transition_kernel(
            position, forward=True
        )
        position_b, momentum_b, accept_prob_b = self.transition_kernel(
            position, forward=False
        )

        # Decide direction uniformly
        batch_size = tf.shape(position)[0]
        forward_mask = tf.cast(tf.random_uniform((batch_size,)) > 0.4,
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
        accept_prob = (
            forward_mask * accept_prob_f + backward_mask * accept_prob_b
        )

        # Accept or reject step
        accept_mask = tf.cast(
            accept_prob > tf.random_uniform(tf.shape(accept_prob)), tf.float32
        )
        reject_mask = 1. - accept_mask

        # Samples after accept / reject step
        position_out = (
            accept_mask[:, None] * position_post
            + reject_mask[:, None] * position
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

        return position_post, momentum_post, accept_prob

    def _forward_lf(self, position, momentum, i):
        """One forward augmented leapfrog step."""
        t = self._get_time(i)
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
        scale, transition, transformed = self.momentum_fn([position, grad, t])
        scale *= 0.5 * self.eps
        transformed *= self.eps
        momentum = (
            momentum * tf.exp(scale)
            - 0.5 * self.eps * (tf.exp(transformed) * grad - translation)
        )

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_forward(self, position, momentum, t, mask, mask_inv):
        """Update x in the forward leapfrog step."""

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
        return position, tf.reduce_sum(mask_inv * scale, axis=1)

    def _update_momentum_backward(self, position, momentum, t):
        """Update v in the backward leapforg step. Inverting the forward
        update."""

        grad = self.grad_potential(position)
        scale, translation, transformed = self.momentum_fn([position, grad, t])
        scale *= -0.5 * self.eps
        transformed *= self.eps
        momentum = (
            tf.exp(scale) * (momentum + 0.5 * self.eps * (tf.exp(transformed) *
                                                          grad - translation))
        )

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_backward(self, position, momentum, t, mask, mask_inv):
        """Update x in the backward leapfrog step. Inverting the forward
        update."""

        scale, translation, transformed = self.position_fn(
            [momentum, mask * position, t]
        )
        scale *= self.eps
        transformed *= self.eps
        position = (
            mask * position + mask_inv * tf.exp(scale) * (position - self.eps *
                                                          (tf.exp(transformed)
                                                           * momentum +
                                                           translation))
        )

        return position, tf.reduce_sum(mask_inv * scale, axis=1)

    def _compute_accept_prob(self, position, momentum, position_post,
                             momentum_post, sumlogdet):
        """Compute the prob of accepting the proposed state given old state."""

        old_hamil = self.hamiltonian(position, momentum)
        new_hamil = self.hamiltonian(position_post, momentum_post)
        prob = tf.exp(tf.minimum(old_hamil - new_hamil + sumlogdet, 0.))

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
            #  idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim // 2]
            #  mask = np.zeros((self.x_dim,))
            #  mask[idx] = 1.
            #  mask = tf.constant(mask, dtype=tf.float32)
            #  self.masks.append(mask[None, :])
            arr = np.arange(self.lattice.num_links)
            idx = np.random.permutation(arr)[:int(self.lattice.num_links // 2)]
            mask = np.zeros((self.lattice.num_links,))
            mask[idx] = 1.
            mask = mask.reshape(self.lattice.link_idxs)
            mask = tf.constant(mask, dtype=tf.float32)
            self.masks.append(mask[None, :])

    def _get_mask(self, i):
        """Get binary masks for i-th augmented leapfrog step."""

        m = self.masks[i]
        return m, 1. - m

    def kinetic(self, v):
        """Compute the kinetic energy."""

        #  flat_v = v.numpy().reshape(-1, self.x_flat_dim)
        axes = np.arange(1, len(v.shape))

        return 0.5 * tf.reduce_sum(v**2, axis=axes)

    def hamiltonian(self, position, momentum):
        """Compute the overall Hamiltonian."""

        return (
            self.potential(position, self.batch_size) + self.kinetic(momentum)
        )

    def grad_potential(self, position, check_numerics=True):
        """Get gradient of potential function at current location."""

        if tf.executing_eagerly():
            grad_fn = tfe.gradients_function(self.potential)
            grad = grad_fn(position, self.batch_size)[0]
            #  grad = (
            #      tfe.gradients_function(self.potential,
            #                             params=[0])(position)[0]
            #  )
        else:
            grad = tf.gradients(self.potential(position), position)[0]
        return grad

# Examples of unnormalized log densities
def get_scg_energy_fn():
    """Get energy function for 2d strongly correlated Gaussian."""

    # Avoid recreating tf.constants on each invocation of gradients
    mu = tf.constant([0., 0.])
    sigma = tf.constant([[50.05, -49.95], [-49.95, 50.05]])
    sigma_inv = tf.matrix_inverse(sigma)

    def energy(x):
        """Unnormalized minus log density of 2d strongly correlated Gaussian"""

        xmmu = x - mu
        return 0.5 * tf.diag_part(
            tf.matmul(tf.matmul(xmmu, sigma_inv), tf.transpose(xmmu))
        )

    return energy, mu, sigma


def get_rw_energy_fn():
    """Get energy function for rough well distribution."""
    # For small eta, the density underlying the rough-well energy is very close
    # to a unit Gaussian; however, the gradient is greatly affected by the
    # small cosine perturbations
    eta = 1e-2
    mu = tf.constant([0., 0.])
    sigma = tf.constant([[1., 0.], [0., 1.]])

    def energy(x):
        ip = tf.reduce_sum(x**2., axis=1)
        return .5 * ip + eta * tf.reduce_sum(tf.cos(x / eta), axis=1)

    return energy, mu, sigma


def get_gmm_energy_fn(means, sigmas, pis):
    """Get energy function for Gaussian Mixture Model."""
    distribution = GMM(means, sigmas, pis)
    return distribution.get_energy_function(), distribution


# Loss function
def compute_loss(dynamics, x, scale=0.1, eps=1e-4):
    """Compute loss defined in Eq. (8) of paper."""

    z = tf.random_normal(tf.shape(x))  # Auxiliary variable
    x_, _, x_accept_prob, x_out = dynamics.apply_transition(x)
    z_, _, z_accept_prob, _ = dynamics.apply_transition(z)

    # Add eps for numerical stability; following released impl
    x_loss = tf.reduce_sum((x - x_)**2, axis=1) * x_accept_prob + eps
    z_loss = tf.reduce_sum((z - z_)**2, axis=1) * z_accept_prob + eps

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
