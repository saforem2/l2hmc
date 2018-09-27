import numpy as np
import tensorflow as tf
import random
from functools import reduce
from scipy.linalg import expm
#  from matrices import GELLMANN_MATRICES, PAULI_MATRICES
from .gauge_generators import generate_SU2, generate_SU3_array


EPS = 0.1

##############################################################################
#  TODO:
#    * Implement U(1) gauge model.
#    * Look at how tensorflow handles gradients for force function in update.
##############################################################################

class GaugeLattice(object):
    """Lattice with Gauge field existing on links."""
    def __init__(self, time_size, space_size, dim, beta, link_type):
        self.time_size = time_size
        self.space_size = space_size
        self.dim = dim
        self.beta = beta
        self.link_type = link_type

        self._init_lattice(link_type)

        self.num_sites = np.cumproduct(self.site_idxs)[-1]
        self.num_links = self.dim * self.num_sites
        self.bases = np.eye(dim, dtype=np.int)

    def _init_lattice(self, link_type):
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

        if self.link_shape != ():
            # Indices for individual sites and links
            self.site_idxs = self.sites.shape[:-2]
            self.link_idxs = self.links.shape[:-2]
        else:
            self.site_idxs = self.sites.shape
            self.link_idxs = self.links.shape

        if link_type == 'SU2':
            for link in self.iter_links():
                self.links[link] = generate_SU2(EPS)

        if link_type == 'SU3':
            self.links = generate_SU3_array(self.num_links // 2,
                                            EPS).reshape(self.links.shape)

        if link_type == 'U1':
            self.links = 2 * np.pi * np.random.rand(*self.link_idxs)

        self.sites_flat = self.sites.flatten()
        self.links_flat = self.links.flatten()



    def _get_random_links(self, link_type=None):
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
            _links = np.zeros(self.links.shape, dtype=tf.complex64)
            for link in self.iter_links():
                _links[link] = generate_SU2(EPS)

        if link_type == 'SU3':
            _links = generate_SU3_array(self.num_links // 2,
                                        EPS).reshape(self.links.shape)

        if link_type == 'U1':
            _links = np.random.rand(*self.links.shape)

        return _links

    def get_links_samples(self, num_samples, link_type=None):
        """Returns `num_samples` randomly initialized links arrays."""
        samples = [
            self._get_random_links(link_type) for _ in range(num_samples)
        ]
        return np.array(samples)

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
        """Returns neighbors of `site`."""
        shape = self.sites.shape
        neighbors = list()
        for i, dim in enumerate(shape):
            e = list(site)
            if site[i] > 0:
                e[i] = e[i] - 1
            else:
                e[i] = dim - 1
            neighbors.append(tuple(e))

            e = list(site)
            if site[i] < dim - 1:
                e[i] = e[i] + 1
            else:
                e[i] = 0
            neighbors.append(tuple(e))
        return neighbors

    def get_random_site(self):
        return tuple([random.randint(0, d-1) for d in self.site_idxs])

    def get_random_link(self):
        return tuple([random.randint(0, d-1) for d in self.link_idxs])

    def get_energy_function(self):
        def fn(links, batch_size):
            return self.total_action(links, batch_size)
        return fn

    def local_action(self, all_links, *links):
        S = 0.0
        for link in links:
            site1 = link[:-1]
            mu = link[-1]
            for nu in range(self.dim):
                if nu != mu:
                    site2 = np.array(site1) - self.bases[nu]
                    plaq1 = self.plaquette_operator(all_links, site1, mu, nu)
                    plaq2 = self.plaquette_operator(all_links, site2, mu, nu)
                    S += (5.0 / 3.0) * (plaq1 + plaq2)
        return S

    def _total_action(self, links=None):
        if links is None:
            links = self.links

        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)

        #  if links.shape[-1] == self.num_links:
        #      links = tf.reshape(links, [-1] + list(self.links.shape))

        S = 0.0
        if self.link_shape != ():
            const = 5.0 / 3.0  # constant multiplicative factor
        else:
            const = 1.

        for site in self.iter_sites():
            for mu in range(self.dim):
                for nu in range(self.dim):
                    if nu > mu:
                        S += const * self.plaquette_operator(links, site,
                                                             mu, nu)
        return np.float32(S)

    def total_action(self, batch, batch_size):
        action_arr = []
        for idx in range(batch_size):
            action_arr.append(self._total_action(batch[idx]))
        return np.array(action_arr, dtype=np.float32)

    def plaquette_operator(self, links, site, mu, nu):
        """Local (counter-clockwise) plaquette operator calculated at `site`

        Args:
            links: Array of link variables (shape = self.links.shape)
            site: Starting point of plaquette loop calculation.
            mu: First direction (0 <= mu <= self.dim - 1)
            nu: Second direction (0 <= nu <= self.dim - 1)
        """
        shape = self.site_idxs

        l1 = tuple(list(np.mod(site, shape)) + [mu])
        l2 = tuple(list(np.mod((site + self.bases[mu]), shape)) + [nu])
        l3 = tuple(list(np.mod((site + self.bases[nu]), shape)) + [mu])
        l4 = tuple(list(np.mod(site, shape)) + [nu])

        if self.link_shape != ():
            prod12 = tf.matmul(links[l1], links[l2])
            prod34 = tf.matmul(tf.transpose(tf.conj(links[l3])),
                               tf.transpose(tf.conj(links[l4])))
            prod1234 = tf.matmul(prod12, prod34)

            return 1.0 * tf.real(tf.trace(prod1234)) / 3.0

        else:
            _sum = links[l1] + links[l2] - links[l3] - links[l4]
            return np.cos(_sum) / 6.  # each site has 6 plaquettes

    def rect_operator(self, links, site, mu, nu):
        shape = self.sites.shape
        site = np.array(site)
        l1 = tuple(list(np.mod(site,shape))+[mu])
        l2 = tuple(list(np.mod((site+self.bases[mu]),shape))+[mu])
        l3 = tuple(list(np.mod((site+2*self.bases[mu]),shape))+[nu])
        l4 = tuple(list(np.mod((site+self.bases[mu]+self.bases[nu]),shape))+[mu])
        l5 = tuple(list(np.mod((site+self.bases[nu]),shape))+[mu])
        l6 = tuple(list(np.mod(site,shape))+[nu])
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
