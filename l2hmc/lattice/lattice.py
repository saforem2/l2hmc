import numpy as np
import random


class IsingLattice(object):
    """Lattice class."""
    def __init__(self, num_sites, dim, coupling=1.):
        self._idxs = dim * (num_sites,)
        self.dim = dim
        self.sites = np.zeros(self._idxs)
        self.num_sites = np.cumprod(self.sites.shape)[-1]

    def iter_sites(self):
        for i in range(self.num_sites):
            indices = list()
            for dim in self.sites.shape:
                indices.append(i % dim)
                i = i // dim
            yield tuple(indices)

    def randomize(self):
        rand_idxs = np.random.randint(2, size=self.sites.shape)
        self.sites[rand_idxs == 1] = 1
        self.sites[rand_idxs == 0] = -1

    def fill_sites(self, val):
        self.sites = val * np.ones(self._idxs)

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
        return tuple([random.randint(0, d-1) for d in self.sites.shape])

    def calc_energy(self):
        """Calculate total energy."""
        energy = 0
        for site in self.iter_sites():
            S = self.sites[site]
            S_nbrs = [S * self.sites[nbr] for nbr in self.get_neighbors(site)]
            energy += sum(S_nbrs)
            #  S_nbrs = 0
            #  for neighbor in self.get_neighbors(site):
            #      S_nbrs += self.sites[neighbor]
            #  energy += -S_nbrs * S
        return energy

    def calc_magnetization(self):
        return np.sum(self.sites)

