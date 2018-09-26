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

        if link_type == 'SU2':
            link_shape = (2, 2)

        if link_type == 'SU3':
            link_shape = (3, 3)

        if link_type == 'U1':
            link_shape = (1,)

        self.link_type = link_type
        self.link_shape = link_shape

        self._sites_shape = tuple([time_size]
                                  + [space_size for _ in range(dim-1)]
                                  + list(link_shape))

        self.sites = np.zeros(self._sites_shape, dtype=np.complex64)
        self.site_idxs = self.sites.shape[:-2]

        self._links_shape = tuple(list(self.site_idxs)
                                  + [dim]
                                  + list(link_shape))

        self.links = np.zeros(self._links_shape, dtype=np.complex64)
        self.link_idxs = self.links.shape[:-2]

        self.dim = dim
        #  self.num_sites = reduce(lambda a, b: a * b, self.site_idxs)
        self.num_sites = np.cumproduct(self.site_idxs)[-1]
        self.num_links = self.dim * self.num_sites
        self.beta = beta
        self.bases = np.eye(dim, dtype=np.int)

        if link_type == 'SU2':
            for link in self.iter_links():
                self.links[link] = generate_SU2(EPS)

        if link_type == 'SU3':
            self.links = generate_SU3_array(self.num_links // 2,
                                            EPS).reshape(self.links.shape)

        if link_type == 'U1':
            for link in self.iter_links():
                self.links[link] = 

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
        return _links

    def get_links_samples(self, num_samples, link_type=None):
        """Returns `num_samples` randomly initialized links arrays."""
        return [self._get_random_links(link_type) for _ in range(num_samples)]

    #  def _random_links(self):
    #      links = np.zeros(
    #          list(self.sites.shape) + [self.dim],
    #          dtype=self.link_type
    #      )
    #      for link in self.iter_links():
    #          links[link] = self.link_type.get_random_element()
    #      return links

    #  def get_samples(self, num_samples):
    #      samples = []
    #      for i in range(num_samples):
    #          links = np.zeros(self.links.shape, dtype=np.complex)
    #          for link in self.iter_links():
    #              links[link] = self.link_type.get_random_element()
    #          samples.append(links)
    #      return np.array(samples)

    def iter_sites(self):
        for i in range(self.num_sites):
            indices = list()
            for dim in self.site_idxs:
                indices.append(i % dim)
                i = i // dim
            yield tuple(indices)

    def iter_links(self):
        for site in self.iter_sites():
            for mu in range(self.dim):
                yield tuple(list(site) + [mu])

    def get_neighbors(self, site):
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

        if links.shape[-1] == self.num_links:
            links = tf.reshape(links, [-1] + list(self.links.shape))

        S = 0.0
        for site in self.iter_sites():
            for mu in range(self.dim):
                for nu in range(self.dim):
                    if nu > mu:
                        S += (5.0 / 3.0) * self.plaquette_operator(links, site,
                                                                   mu, nu)
        return S

    def total_action(self, batch, batch_size):
        action_arr = []
        for idx in range(batch_size):
            action_arr.append(self._total_action(batch[idx]))
        return action_arr

    def plaquette_operator(self, links, site, mu, nu):
        shape = self.site_idxs
        #  site = np.array(site)
        #  shape = self.link_idxs
        #  import pdb
        #  pdb.set_trace()
        l1 = tuple(list(np.mod(site, shape)) + [mu])
        l2 = tuple(list(np.mod((site + self.bases[mu]), shape)) + [nu])
        l3 = tuple(list(np.mod((site + self.bases[nu]), shape)) + [mu])
        l4 = tuple(list(np.mod(site, shape)) + [nu])
        #  if len(l1) == 6 or len(l2) == 6 or len(l3) == 5 or len(l4) == 6:
        #      import pdb
        #      pdb.set_trace()
        #  return 1.0 * np.trace(links[l1]
        #                        * links[l2]
        #                        * links[l3].conjugate().T
        #                        * links[l4].conjugate().T).real / 3.0
        #  try:
        #  prod1 = tf.multiply(links[l2], tf.transpose(tf.conj(links[l3])))
        #  def reshape_link(link):
        #      return tf.reshape(link, self.link_shape)
        #
        #  links[l1] = reshape_link(links[l1])
        #  links[l2] = reshape_link(links[l2])
        #  links[l3] = reshape_link(links[l3])
        #  links[l4] = reshape_link(links[l4])
        #  prod
        prod12 = tf.matmul(links[l1], links[l2])
        prod34 = tf.matmul(tf.transpose(tf.conj(links[l3])),
                           tf.transpose(tf.conj(links[l4])))
        prod1234 = tf.matmul(prod12, prod34)
        try:
            return 1.0 * tf.real(tf.trace(prod1234)) / 3.0
        except ValueError:
            import pdb
            pdb.set_trace()
        #  except:
        #      import pdb
        #      pdb.set_trace()

    def rect_operator(self, links, site, mu, nu):
        shape = self.sites.shape
        site = np.array(site)
        l1 = tuple(list(np.mod(site,shape))+[mu])
        l2 = tuple(list(np.mod((site+self.bases[mu]),shape))+[mu])
        l3 = tuple(list(np.mod((site+2*self.bases[mu]),shape))+[nu])
        l4 = tuple(list(np.mod((site+self.bases[mu]+self.bases[nu]),shape))+[mu])
        l5 = tuple(list(np.mod((site+self.bases[nu]),shape))+[mu])
        l6 = tuple(list(np.mod(site,shape))+[nu])
        return 1.0 * tf.real(tf.trace(links[l1]
                                      * links[l2]
                                      * links[l3]
                                      * tf.transpose(tf.conj(links[l4]))
                                      * tf.transpose(tf.conj(links[l5]))
                                      * tf.transpose(tf.conj(links[l6]))))
        #  return 1.0 * np.trace(links[l1]
        #                        * links[l2]
        #                        * links[l3]
        #                        * links[l4].conjugate().T
        #                        * links[l5].conjugate().T
        #                        * links[l6].conjugate().T).real

    def metropolis_update(self):
        link = self.get_random_link()
        U = self.link_type.get_random_element()

        Si = self.local_action(self.links, link)
        self.links[link] = U * self.links[link]
        Sf = self.local_action(link)

        if np.random.rand() > min(1, np.exp(self.beta*(Sf-Si))):
            self.links[link] = U.conjugate().T * self.links[link]
