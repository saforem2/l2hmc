import numpy as np


class IsingLattice(object):
    """Lattice class."""
    def __init__(self, num_sites, dim):
        idxs = dim * (num_sites,)
        self.sites = np.zeros(idxs)
        self.num_sites = np.cumprod(self.sites.shape)[-1]

    def randomize(self):
        rand_idxs = np.random.randint(2, size=self.sites.shape)
        self.sites[rand_idxs == 1] = 1
        self.sites[rand_idxs == 0] = -1

