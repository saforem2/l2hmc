import numpy as np
import tensorflow as tf
import random
from functools import reduce
from scipy.linalg import expm
from scipy.special import i0, i1
#  from matrices import GELLMANN_MATRICES, PAULI_MATRICES
from .gauge_generators import generate_SU2, generate_SU3, generate_SU3_array

from HMC.hmc import HMC

import tensorflow.contrib.eager as tfe

EPS = 0.1

NUM_SAMPLES = 500
PHASE_MEAN = 0
PHASE_SIGMA = 0.5  # for phases within +/- π / 6 ~ 0.5
PHASE_SAMPLES = np.random.normal(PHASE_MEAN, PHASE_SIGMA, NUM_SAMPLES // 2)

# the random phases must come in +/- pairs to ensure ergodicity
RANDOM_PHASES = np.append(PHASE_SAMPLES, -PHASE_SAMPLES)

###############################################################################
#                      GLOBAL VARIABLES
# ------------------------------------------------------------------------------
NUM_CONFIGS_PER_SAMPLE = 10000
NUM_SAMPLES = 25
NUM_EQ_CONFIGS = 20000
NUM_CONFIGS = NUM_CONFIGS_PER_SAMPLE * NUM_SAMPLES
###############################################################################

def u1_plaq_exact(beta):
    return i1(beta) / i0(beta)

def pbc(tup, shape):
    return list(np.mod(tup, shape))

def mat_adj(mat):
    return tf.transpose(tf.conj(mat))  # conjugate transpose

class GaugeLattice(object):
    """Lattice with Gauge field existing on links."""
    def __init__(self, 
                 time_size, 
                 space_size, 
                 dim, 
                 beta, 
                 link_type,
                 num_samples=None,
                 rand=False):
        """
            Args:
                time_size (int): Temporal extent of lattice.
                space_size (int): Spatial extent of lattice.
                dim (int): Dimensionality
                link_type (str): 
                    String representing the type of gauge group for the link
                    variables. Must be either 'U1', 'SU2', or 'SU3'
        """
        assert link_type.upper() in ['U1', 'SU2', 'SU3'], ("Invalid link_type."
                                                           "Possible values:"
                                                           "'U1', 'SU2', 'SU3'")
        self.time_size = time_size
        self.space_size = space_size
        self.dim = dim
        self.beta = beta
        self.link_type = link_type
        self.link_shape = None

        self._init_lattice(link_type, rand)
        self.samples = None

        self.num_sites = np.cumproduct(self.site_idxs)[-1]
        self.num_links = int(self.dim * self.num_sites)
        self.bases = np.eye(dim, dtype=np.int)

        if num_samples is not None:
            #  Create `num_samples` randomized instances of links array
            self.num_samples = num_samples
            self.samples = self.get_links_samples(num_samples, rand=rand,
                                                  link_type=self.link_type)
            self.samples[0] = self.links

    def _init_lattice(self, link_type, rand):
        """Initialize lattice by creating self.sites and sites.links variables.
        
        Link variables are randomly initialized to elements in their respective
        gauge group.
        """

        if link_type == 'SU2':
            self.link_shape = (2, 2)

        if link_type == 'SU3':
            self.link_shape = (3, 3)

        if link_type == 'U1':
            self.link_shape = ()

        sites_shape = tuple(
            [self.time_size]
            + [self.space_size for _ in range(self.dim-1)]
            + list(self.link_shape)
        )

        links_shape = tuple(
            [self.time_size]
            + [self.space_size for _ in range(self.dim-1)]
            + [self.dim]
            + list(self.link_shape)
        )

        self.sites = np.zeros(sites_shape, dtype=np.complex64)
        self.links = np.zeros(links_shape, dtype=np.complex64)
        #  self.links = self._generate_links(rand, link_type)
        self.num_sites = np.cumprod(self.sites.shape)[-1]
        self.num_links = self.num_sites * self.dim

        #  if self.link_shape != ():
        if self.link_type != 'U1':
            # Indices for individual sites and links
            self.site_idxs = self.sites.shape[:-2]
            self.link_idxs = self.links.shape[:-2]
            self.links = np.zeros(links_shape, dtype=np.complex64)
        else:
            self.site_idxs = self.sites.shape
            self.link_idxs = self.links.shape
            if rand:
                self.links = np.array(np.random.uniform(0, 2*np.pi,
                                                        links_shape),
                                      dtype=np.float32)
            else:
                self.links = np.zeros(links_shape, dtype=np.float32)
            #  self.links = 2 * np.pi * np.random.rand(*self.link_idxs)

        self.sites_flat = self.sites.flatten()
        self.links_flat = self.links.flatten()

    def _generate_links(self, rand=False, link_type=None):
        """Method for obtaning an array of randomly initialized link variables.
            Args:
                link_type (str): 
                    Specifies the gauge group to be used on the links.

            Returns:
                _links (np.ndarray):
                    Array of the same shape as self.links.shape, containing
                    randomly initialized link variables.
        """
        if link_type is None:
            link_type = self.link_type

        if link_type == 'SU2':
            links = np.zeros(self.links.shape, dtype=np.complex64)
            if rand:
                for link in self.iter_links():
                    links[link] = generate_SU2(EPS)

        if link_type == 'SU3':
            links = np.zeros(self.links.shape, dtype=np.complex64)
            if rand:
                for link in self.iter_links():
                    links[link] = generate_SU3(EPS)

        if link_type == 'U1':
            if rand:
                links = 2 * np.pi * np.random.rand(*self.links.shape)
            else:
                links = np.zeros(self.links.shape)
        return links

    def get_links_samples(self, num_samples, rand=False, link_type=None):
        """Return `num_samples` randomly initialized links arrays."""
        samples = np.array([
            self._generate_links(rand, link_type) for _ in range(num_samples)
        ])
        return samples

    def iter_sites(self):
        """Iterator for looping over sites."""
        for i in range(self.num_sites):
            indices = list()
            for dim in self.site_idxs:
                indices.append(i % dim)
                i = i // dim
            yield tuple(indices)

    def iter_links(self):
        """Iterator for looping over links."""
        for site in self.iter_sites():
            for u in range(self.dim):
                yield tuple(list(site) + [u])

    def get_neighbors(self, site):
        """Return neighbors of `site`."""
        shape = self.sites.shape
        neighbors = list()
        for i, dim in enumerate(shape):
            nbr = list(site)
            if site[i] > 0:
                nbr[i] = nbr[i] - 1
            else:
                nbr[i] = dim - 1
            neighbors.append(tuple(nbr))

            nbr = list(site)
            if site[i] < dim - 1:
                nbr[i] = nbr[i] + 1
            else:
                nbr[i] = 0
            neighbors.append(tuple(nbr))
        return neighbors

    def get_random_site(self):
        """Return indices of randomly chosen site."""
        return tuple([random.randint(0, d-1) for d in self.site_idxs])

    def get_random_link(self):
        """Return inidices of randomly chosen link."""
        return tuple([random.randint(0, d-1) for d in self.link_idxs])

    def get_link(self, site, direction, shape, links=None):
        """Returns the value of the link variable located at site + direction."""
        if links is None:
            links = self.links
        return links[tuple(pbc(site, shape) + [direction])]

    def get_energy_function(self, samples=None):
        """Returns function object used for calculating the energy (action)."""
        if samples is None:
            def fn(links):
                return self._total_action(links)
        else:
            def fn(samples):
                return self.total_action(samples)
        return fn

    def _average_plaquette(self, links=None):
        """Computes the average plaquette of a particular lattice of links."""
        if links is None:
            links = self.links
            links = tf.reshape(links, self.links.shape)
            #links = links.reshape(self.links.shape)

        num_plaquettes = self.time_size * self.space_size
        plaquette_sum = 0.
        for site in self.iter_sites():
            for u in range(self.dim):
                for v in range(self.dim):
                    if v > u:
                        plaq = self.plaquette_operator(site, u, v, links)
                        plaquette_sum += self._action_op(plaq)

        return plaquette_sum / num_plaquettes

    def average_plaquette(self, samples=None):
        """Calculate the average plaquette for each sample in samples."""
        if samples is None:
            if self.samples is None:
                return self._average_plaquette()
            samples = self.samples

        return [self._average_plaquette(sample) for sample in samples]

    def local_action(self, *links, all_links):
        """Compute local action (internal energy) of a collection of `links`
        that belong to lattice.

        Args:
            *links (array-like):
                Collection of links over which to calculate the local action.
            all_links (array-like):
                Links array, shape = self.links.shape 
        """
        #  assert all_links.shape != self.links.shape, ("`all_links` must have"
        #                                               " the same shape as"
        #                                               " self.links.shape.")

        S = 0.0
        for link in links:
            site1 = link[:-1]
            u = link[-1]
            for v in range(self.dim):
                if v != u:
                    site2 = np.array(site1) - self.bases[v]
                    plaq1 = self.plaquette_operator(site1, u, v, all_links)
                    plaq2 = self.plaquette_operator(site2, u, v, all_links)
                    S += (plaq1 + plaq2)
        return S

    def _total_action(self, links=None):
        """
        Computes the total action of an individual lattice by summing the
        internal energy of each plaquette over all plaquettes.

        * For SU(N) (N = 2, 3), the action of a single plaquette is
        calculated as:
            Sp = 1 - Re{Tr(Up)}, where Up is the plaquette_operator defined as
            the product of the gauge fields around an elementary plaquette.

        * For U(1), the action of a sinigle plaquette is calculated as:
            Sp = 1 - cos(Qp), where Qp is the plaquette operator defined as the
            sum of the angles (phases) around an elementary plaquette.
        """
        if links is None:
            links = self.links
        if links.shape != self.links.shape:
            links = tf.reshape(links, self.links.shape)

        action = 0.0
        for site in self.iter_sites():
            for u in range(self.dim):
                for v in range(self.dim):
                    if v > u:
                        plaq = self.plaquette_operator(site, u, v, links)
                        action += 1 - self._action_op(plaq)
                        #  S += 1 - const * plaq
        return self.beta * action #/ self.num_sites

    def total_action(self, samples=None):
        """Return the total action (sum over all plaquettes) for each sample in
        samples, at inverse coupling strength `self.beta`. 

            Args:
                samples (array-like):
                    Array of `links` arrays, each one representing the links of
                    an individual lattice.  NOTE: If samples is None, only a
                    single `GaugeLattice` was instantiated during the __init__
                    method, so this will return the total action of that single
                    lattice.
            Returns:
                _ (float or list of floats): 
                    If samples is None, returns action of instantiated lattice
                    as a float. Otherwise, returns list containing the action
                    of each sample in samples.
        """
        if samples is None:
            if self.samples is None:
                return self._total_action()
            samples = self.samples

        return [self._total_action(sample) for sample in samples]

    def _local_grad_action(self, site, u, links=None):
        """Compute the local gradient of the action with respect to the link
        variables for a single lattice instance."""
        if links is None:
            links = self.links

        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)
            #  links = links.reshape(self.links.shape)

        grad = np.float32(0.0)
        shape = self.site_idxs
        for v in range(self.dim):
            if v != u:
                site2 = np.mod((site - self.bases[v]), shape)

                plaq1 = self.plaquette_operator(site, u, v, links)
                plaq2 = self.plaquette_operator(site2, u, v, links)
                grad += (self._grad_action_op(plaq1)
                             - self._grad_action_op(plaq2))


                #  shifted_site = np.mod((site - self.bases[v]), shape)
                #  plaq1 = self.plaquette_operator(links, site, u, v)
                #  plaq2 = self.plaquette_operator(links, shifted_site, u, v)
                #  force += np.sin(plaq2) - np.sin(plaq1)
        return self.beta * grad

    def _grad_action(self, links=None):
        """Compute the gradient of the action for each array of links in
        `self.samples`."""
        if links is None:
            links = self.links

        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)

        grad_arr = np.zeros(links.shape)
        for site in self.iter_sites():
            for u in range(self.dim):
                grad_arr[site][u] = self._local_grad_action(site, u)

        return grad_arr.flatten()

    def grad_action(self, samples=None):
        """Return the gradient of the action for each sample in samples, at
        inverse coupling strength `self.beta`.

        NOTE: If samples is None, only a single `GaugeLattice` was instantiated
        during the __init__ method, so this will return the gradient of the
        action of that single lattice instance.
        """
        if samples is None:
            return self._grad_action()
        return [
            self._grad_action(sample) for sample in samples
        ]

    def _action_op(self, plaq):
        """Operator used in calculating the action."""
        if self.link_type == 'U1':
            #  return np.cos(plaq)
            return tf.math.cos(plaq)
        return 1.0 * tf.real(tf.trace(plaq)) / self.link_shape[0]

    def _grad_action_op(self, plaq):
        """Operator used in calculating the gradient of the action."""
        if self.link_type == 'U1':
            return np.sin(plaq)
        return tf.imag(tf.trace(plaq)) / self.link_shape[0]

    def _link_staple_op(self, link, staple):
        """Operator used in calculating the change in the action caused by
        updating an individual `link`."""
        if self.link_type == 'U1':
            return np.cos(link + staple)
        return tf.matmul(link, staple)

    def plaquette_operator(self, site, u, v, links=None):
        """Local (counter-clockwise) plaquette operator calculated at `site`
            Args:
                site (tuple): 
                    Starting point (lower left site) of plaquette loop
                    calculation.
                u (int): 
                    First direction (0 <= u <= self.dim - 1)
                v (int): 
                    Second direction (0 <= v <= self.dim - 1)
                links (array-like): 
                    Array of link variables (shape = self.links.shape). If none
                    is provided, self.links will be used.
        """
        if links is None:
            links = self.links

        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)

        shape = self.site_idxs

        l1 = self.get_link(site, u, shape, links)                  # U(x; u)
        l2 = self.get_link(site + self.bases[u], v, shape, links)  # U(x + u; v)
        l3 = self.get_link(site + self.bases[v], u, shape, links)  # U(x + v; u)
        l4 = self.get_link(site, v, shape, links)                  # U(x; v)

        if self.link_type == 'U1':
            return l1 + l2 - l3 - l4
        elif self.link_type in ['SU2', 'SU3']:
            prod = tf.matmul(l1, l2)
            prod = tf.matmul(prod, mat_adj(l3))
            prod = tf.matmul(prod, mat_adj(l4))
            #  prod = tf.matmul(links[l1], links[l2])
            #  prod = tf.matmul(prod, tf.transpose(tf.conj(links[l3])))
            #  prod = tf.matmul(prod, tf.transpose(tf.conj(links[l4])))
            return prod
        else:
            #  return 1.0 * tf.real(tf.trace(prod1234)) / 3.0, prod1234
            #  return np.cos(_sum)., _sum  # each site has 6 plaquettes
            raise AttributeError('Link type must be one of `U1`, `SU2`, `SU3`')

    def _get_staples(self, site, u, links=None):
        """Calculates each of the `staples` for the link variable located at 
        site + u."""
        if links is None:
            links = self.links
        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)

        shape = self.site_idxs

        staples = []
        for v in range(self.dim):  # u, v instead of mu, nu for readability
            if v != u:
                l1 = self.get_link(site + self.bases[u], v, shape, links)
                l2 = self.get_link(site + self.bases[v], u, shape, links)
                l3 = self.get_link(site, v, shape, links)

                l4 = self.get_link(site + self.bases[u] - self.bases[v], v,
                                   shape, links)
                l5 = self.get_link(site - self.bases[v], u, shape, links)
                l6 = self.get_link(site - self.bases[v], v, shape, links)

                if self.link_type == 'U1':
                    _sum1 = l1 - l2 - l3
                    _sum2 = -l4 -l5 + l6

                elif self.link_type in ['SU2', 'SU3']:
                    prod1 = tf.matmul(l1, mat_adj(l2))
                    prod1 = tf.matmul(prod1, mat_adj(l3))

                    prod2 = tf.matmul(mat_adj(l3), mat_adj(l4))
                    prod2 = tf.matmul(prod2, mat_adj(l6))

                    _sum = prod1 + prod2

                #_arr = [_sum1, _sum2]
                staples.append(_sum1)
                staples.append(_sum2)

        return staples

    def rect_operator(self, site, u, v, links=None):
        if links is None:
            links = self.links

        shape = self.sites.shape

        #  site = np.array(site)
        l1 = tuple(pbc(site) + [u])  #pylint: ignore invalid-name
        l2 = tuple(pbc(site + self.bases[u]) + [u])
        l3 = tuple(pbc(site + 2 * self.bases[u]) + [v])
        l4 = tuple(pbc(site + self.bases[u]+self.bases[v]) + [u])
        l5 = tuple(pbc(site + self.bases[v]) + [u])
        l6 = tuple(pbc(site) + [v])

        if self.link_shape != ():
            return 1.0 * tf.real(tf.trace(links[l1]
                                          * links[l2]
                                          * links[l3]
                                          * tf.transpose(tf.conj(links[l4]))
                                          * tf.transpose(tf.conj(links[l5]))
                                          * tf.transpose(tf.conj(links[l6]))))
        else:
            return (links[l1] + links[l2] + links[l3]
                    - links[l4] - links[l5] - links[l6])

    def _update_link(self, site, d, links=None):
        """Update the link located at site + d using Metropolis-Hastings
        accept/reject."""
        if links is None:
            links = self.links

        if links.shape != self.links.shape:
            links = tf.reshape(links, self.links.shape)

        shape = self.site_idxs

        staples = self._get_staples(site, d, links)

        current_link = self.get_link(site, d, shape, links)
        proposed_link = current_link + np.random.choice(RANDOM_PHASES)

        minus_current_action = np.sum(
            [np.cos(current_link + s) for s in staples]
        )
        minus_proposed_action = np.sum(
            [np.cos(proposed_link + s) for s in staples]
        )

        # note that if the proposed action is smaller than the current action,
        # prob > 1 and we accept the new link
        prob = min(1, np.exp(self.beta * (minus_proposed_action
                                          - minus_current_action)))
        accept = 0
        if np.random.uniform() < prob:
            self.links[tuple(pbc(site, shape) + [d])] = proposed_link
            accept = 1
        return accept

    def run_metropolis(self, links=None):
        """Run the MCMC simulation using Metropolis-Hastings accept/reject. """
        if links is None:
            links = self.links
        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)
        # relax the initial configuration
        eq_steps = 1000
        for step in range(eq_steps):
            for site in self.iter_sites():
                for d in range(self.dim):
                    _ = self._update_link(self, site, d)

        num_acceptances = 0  # keep track of acceptance rate
        for step in range(10000):
            for site in self.iter_sites():
                for d in range(self.dim):
                    num_acceptances += self._update_link(self, site, d)

