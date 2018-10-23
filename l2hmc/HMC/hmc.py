import numpy as np
import tensorflow as tf

import tensorflow.contrib.eager as tfe

class HMC(object):
    """Generic HMC object for implementing Hybrid Monte Carlo algorithm."""
    def __init__(self, position_init, step_size, n_leapfrog_steps, 
                 potential_fn, grad_potential_fn=None,):
        """Initialize HMC object.

        Args:
            position_init (aray-like, tensor, or object): 
                Quantity of interest in HMC simulation, representing the
                variable to be updated.
            step_size: 
                Scalar step size or array of step sizes for the leapfrog
                integrator. Larger step sizes lead to faster progress, but
                too-large step sizes make rejection exponentially more likely.
                When possible, it's often helpful to match per-variable step
                sizes to the standard deviations of the target distribution in
                each variable.
            n_leapfrog_steps: 
                Integer number of steps to run the leapfrog integrator for.
                Total progress per HMC step is roughly proportional to
                step_size * n_leapfrog_steps.
        """
        self.position_init = position_init
        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
        self.potential = potential_fn
        self.grad_potential_fn = grad_potential_fn
        self._construct_time()

    def apply_transition(self, position):
        """Propose a new state and perform accept/reject step."""
        if not isinstance(position, np.ndarray):
            position = np.array(position)

        #  momentum = tf.random_normal(tf.shape(position))
        momentum = np.random.randn(*position.shape)
        position_post, momentum_post = position, momentum
        # Apply leapfrog steps
        for i in range(self.n_leapfrog_steps):
            leapfrog_out = self._leapfrog_fn(position_post, momentum_post, i)
            position_post, momentum_post = leapfrog_out

        old_hamil = self.hamiltonian(position, momentum)
        new_hamil = self.hamiltonian(position_post, momentum_post)
        prob = np.exp(np.minimum(old_hamil - new_hamil, 0.))
        #  prob = np.minimum(np.exp(old_hamil - new_hamil), 1)
        if np.random.uniform() <= prob:
            return position_post, momentum_post, prob
        return position, momentum, prob
        #  accept_prob = self._compute_accept_prob(position, momentum,
        #                                          position_post, momentum_post)
        #
        #  return position_post, momentum_post, accept_prob

    def _leapfrog_fn(self, position, momentum, i):
        """One leapfrog step."""
        t = self._get_time(i)  # pylint: ignore-invalid-name
        momentum = self._update_momentum(position, momentum, t)
        position = self._update_position(position, momentum, t)
        momentum = self._update_momentum(position, momentum, t)

        return position, momentum

    def _update_momentum(self, position, momentum, t):
        """Update momentum in the leapfrog step."""
        if self.grad_potential_fn is None:
            grad = self._grad_potential(position)
        else:
            grad = self.grad_potential_fn(position)
        momentum_out = momentum - 0.5 * self.step_size * grad

        return momentum_out

    def _update_position(self, position, momentum, t):
        """Update position in the leapfrog step."""
        return position + self.step_size * momentum

    def _compute_accept_prob(self, position, momentum,
                             position_post, momentum_post):
        """Compute the prob of accepting the proposed state given old state."""
        old_hamil = self.hamiltonian(position, momentum)
        new_hamil = self.hamiltonian(position_post, momentum_post)
        prob = np.exp(np.minimum(old_hamil - new_hamil, 0.))

        # Ensure mathematical stability as well as correct gradients
        return np.where(np.isfinite(prob), prob, np.zeros_like(prob))

    def _construct_time(self):
        """Convert leapfrog step index into sinusoidal time."""
        self.ts = []
        for i in range(self.n_leapfrog_steps):
            #  t = tf.constant(
            t = [np.cos(2 * np.pi * i / self.n_leapfrog_steps),
                 np.sin(2 * np.pi * i / self.n_leapfrog_steps)]
            self.ts.append(t)
            #  self.ts.append(t[None, :])
        self.ts = np.array(self.ts)

    def _get_time(self, i):
        """Get sinusoidal time for i-th leapfrog step."""
        return self.ts[i]

    def kinetic(self, v):
        """Compute the kinetic energy."""
        #return 0.5 * tf.reduce_sum(v**2)
        return 0.5 * np.sum(np.square(v))
        #  return 0.5 * np.sum(v**2, axis=1)

    def hamiltonian(self, position, momentum):
        """Compute the overall Hamiltonian."""
        return self.potential(position) + self.kinetic(momentum)

    def _grad_potential(self, position, check_numerics=True):
        """Get gradient of potential function at current position."""
        if tf.executing_eagerly():
            grad = tfe.gradients_function(self.potential)(position)[0]
        else:
            grad = tf.gradients(self.potential(position), position)[0]

        return grad
