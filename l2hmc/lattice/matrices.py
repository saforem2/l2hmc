import numpy as np

DIRAC_MATRICES = np.array([
    np.matrix([
        [+1, 0, 0, 0],
        [0, +1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, +1],
        [0, 0, +1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, -1j],
        [0, 0, +1j, 0],
        [0, +1j, 0, 0],
        [-1j, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, +1, 0],
        [0, 0, 0, -1],
        [-1, 0, 0, 0],
        [0, +1, 0, 0],
    ], dtype=np.complex)
])


CHIRAL_DIRAC_MATRICES = np.array([
    np.matrix([
        [0, 0, +1, 0],
        [0, 0, 0, +1],
        [+1, 0, 0, 0],
        [0, +1, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, +1j],
        [0, 0, +1j, 0],
        [0, -1j, 0, 0],
        [-1j, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, -1j],
        [0, 0, +1j, 0],
        [0, +1j, 0, 0],
        [-1j, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, +1j, 0],
        [0, 0, 0, -1j],
        [-1j, 0, 0, 0],
        [0, 1j, 0, 0],
    ], dtype=np.complex),
])

GAMMA_5 = np.matrix([
    [0, 0, +1, 0],
    [0, 0, 0, +1],
    [+1, 0, 0, 0],
    [0, +1, 0, 0],
], dtype=np.complex)


CHIRAL_GAMMA_5 = np.matrix([
    [+1, 0, 0, 0],
    [0, +1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1],
], dtype=np.complex)

ETA = np.matrix([
    [-1, 0, 0, 0],
    [0, +1, 0, 0],
    [0, 0, +1, 0],
    [0, 0, 0, +1],
], dtype=np.int8)
