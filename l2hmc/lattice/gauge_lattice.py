import numpy as np
import tensorflow as tf
import random
from functools import reduce
from scipy.linalg import expm
#  from matrices import GELLMANN_MATRICES, PAULI_MATRICES
from .gauge_generators import generate_SU2, generate_SU3, generate_SU3_array

from HMC.hmc import HMC

import tensorflow.contrib.eager as tfe

EPS = 0.1

NUM_SAMPLES = 500
PHASE_MEAN = 0
PHASE_SIGMA = 0.5
PHASE_SAMPLES = np.random.normal(PHASE_MEAN, PHASE_SIGMA, NUM_SAMPLES // 2)

RANDOM_PHASES = np.append(PHASE_SAMPLES, -PHASE_SAMPLES)



##############################################################################
#  TODO:
#    * Implement U(1) gauge model.
#    * Look at how tensorflow handles gradients for force function in update.
##############################################################################
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
            #  self.links = np.zeros(links_shape, dtype=np.float32)
            self.links = 2 * np.pi * np.random.rand(*self.link_idxs)

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
            for mu in range(self.dim):
                yield tuple(list(site) + [mu])

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

    def get_energy_function(self):
        """Return function object used for calculating the energy (action)."""
        def fn(samples):
            return self.total_action(samples)
        return fn

    def _action_op(self, plaq):
        """Operator used in calculating the action."""
        if self.link_type == 'U1':
            return np.cos(plaq)
        return 1.0 * tf.real(tf.trace(plaq)) / self.link_shape[0]

    def _grad_action_op(self, plaq):
        """Operator used in calculating the gradient of the action."""
        if self.link_type == 'U1':
            return np.sin(plaq)
        return tf.imag(tf.trace(plaq)) / self.link_shape[0]

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
            mu = link[-1]
            for nu in range(self.dim):
                if nu != mu:
                    site2 = np.array(site1) - self.bases[nu]
                    plaq1 = self.plaquette_operator(site1, mu, nu, all_links)
                    plaq2 = self.plaquette_operator(site2, mu, nu, all_links)
                    S += (plaq1 + plaq2)
        return S

    def _staple(self, site, mu, links=None):
        """Calculate the `staple` at `site` in direction `mu`. """
        if links is None:
            links = self.links

        shape = self.site_idxs

        def pbc(tup):
            return list(np.mod(tup, shape))

        for nu in range(self.dim):
            l1 = tuple(pbc(site + self.bases[mu]) + [nu])
            l2 = tuple(pbc(site + self.bases[nu]) + [mu])
            l3 = tuple(pbc(site) + [nu])
            l4 = tuple(pbc(site + self.bases[mu] - self.bases[nu]) + [nu])
            l5 = tuple(pbc(site - self.bases[nu]) + [mu])
            l5 = tuple(pbc(site - self.bases[nu]) + [mu])
            l6 = tuple(pbc(site - self.bases[nu]) + [nu])

            #  l1 = tuple(list(np.mod( (site + self.bases[mu]), shape) + [nu]))
            #  l2 = tuple(list(np.mod((site + self.bases[nu]), shape)) + [mu])
            #  l3 = tuple(list(np.mod(site, shape)) + [nu])
            #  l4 = tuple(list(np.mod((site + self.bases[mu] - self.bases[nu]),
            #                         shape)) + [nu])
            #  l5 = tuple(list(np.mod((site -self.bases[nu]), shape)) + [mu])
            #  l6 = tuple(list(np.mod((site - self.bases[nu]), shape)) + [nu])
        def mat_adj(mat):
            return tf.transpose(tf.conj(mat))

        if self.link_type == 'U1':
            _sum = (links[l1] - links[l2] - links[l3]
                    - links[l4] - links[l5] - links[l6])
            return _sum
        elif self.link_type in ['SU2', 'SU3']:
            prod1 = tf.matmul(links[l1], mat_adj(links[l2]))
            prod1 = tf.matmul(prod1, mat_adj(links[l3]))

            prod2 = tf.matmul(mat_adj(links[l3]), mat_adj(links[l4]))
            prod2 = tf.matmul(prod2, mat_adj(links[l6]))

            #  prod = tf.matmul(links[l1], tf.tranpose(tf.conj(links[l2])))
            #  prod12 = tf.matmul(links[l1], tf.transpose(tf.conj(links[l2])))
            #  prod123 = tf.matmul(prod12, tf.transpose(tf.conj(links[l3])))
            #
            #  prod45 = tf.matmul(tf.transpose(tf.conj(links[l3])),
            #                     tf.transpose(tf.conj(links[l4])))
            #  prod456 = tf.matmul(prod45, tf.transpose(tf.conj(links[l6])))
            #  return prod123 + prod456
            return prod1 + prod2

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

        action = 0.0
        if self.link_shape != ():
            const = 5.0 / self.link_shape[0]  # constant mult. factor for SU(N)
        else:
            const = 1. # factor for U(1)

        for site in self.iter_sites():
            for mu in range(self.dim):
                for nu in range(self.dim):
                    if nu > mu:
                        plaq = self.plaquette_operator(site, mu, nu, links)
                        action += 1 - const * self._action_op(plaq)
                        #  S += 1 - const * plaq
        return self.beta * action / self.num_sites

    def total_action(self, samples=None):
        """Return the total action (sum over all plaquettes) for each sample in
        samples, at inverse coupling strength `self.beta`. 

        Args:
            samples (array-like):
                Array of `links` arrays, each one representing the links of an
                individual lattice.
                NOTE: If samples is None, only a single `GaugeLattice` was
                instantiated during the __init__ method, so this will return
                the total action of that single lattice.
        Returns:
            _ (float or list of floats): 
                If samples is None, returns action of instantiated lattice as a
                float. Otherwise, returns list containing the action of each
                sample in samples.
        """
        if samples is None:
            if self.samples is None:
                return self._total_action()
            samples = self.samples

        return [
            self._total_action(sample) for sample in samples
        ]

    def _local_grad_action(self, site, mu, links=None):
        """Compute the local gradient of the action with respect to the link
        variables for a single lattice instance."""
        if links is None:
            links = self.links

        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)
            #  links = links.reshape(self.links.shape)

        grad = np.float32(0.0)
        shape = self.site_idxs
        for nu in range(self.dim):
            if nu != mu:
                site2 = np.mod((site - self.bases[nu]), shape)

                plaq1 = self.plaquette_operator(site, mu, nu, links)
                plaq2 = self.plaquette_operator(site2, mu, nu, links)
                grad += (self._grad_action_op(plaq1)
                             - self._grad_action_op(plaq2))

                #  staple = self._staple(site, mu, nu, links)

                #  shifted_site = np.mod((site - self.bases[nu]), shape)
                #  plaq1 = self.plaquette_operator(links, site, mu, nu)
                #  plaq2 = self.plaquette_operator(links, shifted_site, mu, nu)
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
            for mu in range(self.dim):
                grad_arr[site][mu] = self._local_grad_action(site, mu)

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

    def plaquette_operator(self, site, mu, nu, links=None):
        """Local (counter-clockwise) plaquette operator calculated at `site`
            Args:
                site (tuple): 
                    Starting point of plaquette loop calculation.
                mu (int): 
                    First direction (0 <= mu <= self.dim - 1)
                nu (int): 
                    Second direction (0 <= nu <= self.dim - 1)
                links (array-like): 
                    Array of link variables (shape = self.links.shape)
        """
        if links is None:
            links = self.links

        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)

        shape = self.site_idxs

        def pbc(tup):
            return list(np.mod(tup, shape))

        l1 = tuple(pbc(site) + [mu])  # U_mu(x)
        l2 = tuple(pbc(site + self.bases[mu]) + [nu])  # U_nu(x + mu)
        l3 = tuple(pbc(site + self.bases[nu]) + [mu])  # U_mu(x + nu)
        l4 = tuple(pbc(site) + [nu])  # U_nu(x)
        #  import pdb
        #  pdb.set_trace()

        if self.link_type == 'U1':
            return links[l1] + links[l2] - links[l3] - links[l4]
        elif self.link_type in ['SU2', 'SU3']:
            prod = tf.matmul(links[l1], links[l2])
            prod = tf.matmul(prod, tf.transpose(tf.conj(links[l3])))
            prod = tf.matmul(prod, tf.transpose(tf.conj(links[l4])))
            #  prod12 = tf.matmul(links[l1], links[l2])
            #  prod34 = tf.matmul(tf.transpose(tf.conj(links[l3])),
            #                     tf.transpose(tf.conj(links[l4])))
            #  prod1234 = tf.matmul(prod12, prod34)

            #  return prod1234
            return prod
        else:
            #  return 1.0 * tf.real(tf.trace(prod1234)) / 3.0, prod1234
            #  return np.cos(_sum)., _sum  # each site has 6 plaquettes
            raise AttributeError('Link type must be one of `U1`, `SU2`, `SU3`')

    def rect_operator(self, site, mu, nu, links=None):
        if links is None:
            links = self.links

        shape = self.sites.shape

        def pbc(tup):
            return list(np.mod(tup), shape)
        #  site = np.array(site)
        l1 = tuple(pbc(site) + [mu])
        l2 = tuple(pbc(site + self.bases[mu]) + [mu])
        l3 = tuple(pbc(site + 2 * self.bases[mu]) + [nu])
        l4 = tuple(pbc(site + self.bases[mu]+self.bases[nu]) + [mu])
        l5 = tuple(pbc(site + self.bases[nu]) + [mu])
        l6 = tuple(pbc(site) + [nu])

        #  l1 = tuple(list(np.mod(site,shape))+[mu])
        #  l2 = tuple(list(np.mod((site+self.bases[mu]),shape))+[mu])
        #  l3 = tuple(list(np.mod((site+2*self.bases[mu]),shape))+[nu])
        #  l4 = tuple(list(np.mod((site+self.bases[mu]+self.bases[nu]),shape))+[mu])
        #  l5 = tuple(list(np.mod((site+self.bases[nu]),shape))+[mu])
        #  l6 = tuple(list(np.mod(site,shape))+[nu])
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

    def metropolis_update(self):
        link = self.get_random_link()
        U = self.link_type.get_random_element()

        Si = self.local_action(self.links, link)
        self.links[link] = U * self.links[link]
        Sf = self.local_action(link)

        if np.random.rand() > min(1, np.exp(self.beta*(Sf-Si))):
            self.links[link] = U.conjugate().T * self.links[link]
